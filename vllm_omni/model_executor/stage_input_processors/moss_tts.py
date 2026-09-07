# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stage input processors: MOSS-TTS talker (Stage 0) → codec (Stage 1)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TokensPrompt as OmniTokensPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

logger = init_logger(__name__)

_MOSS_AUDIO_PAD_CODE = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull audio codes from a Stage-0 OmniOutput or raw tensor."""
    if stage_output is None:
        return None

    # OmniOutput
    mm = getattr(stage_output, "multimodal_outputs", None)
    if mm is not None:
        codes_dict = mm.get("codes", {})
        if isinstance(codes_dict, dict):
            ac = codes_dict.get("audio")
            if isinstance(ac, torch.Tensor):
                return ac

    return None


def _moss_connector_extra(transfer_manager: Any) -> Mapping[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    config = getattr(connector, "config", {}) or {}
    config = config.get("extra", config) if isinstance(config, Mapping) else {}
    return config if isinstance(config, Mapping) else {}


# ---------------------------------------------------------------------------
# Non-streaming (sync): called once after Stage 0 finishes
# ---------------------------------------------------------------------------


def talker2codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Convert all talker codes to a single Stage-1 token sequence.

    Stage 0 output contains ``codes["audio"]`` shaped ``(T, NQ)`` where T is
    the number of generated audio frames and NQ is n_vq.  We flatten to
    ``[NQ * T]`` as the Stage-1 ``input_ids`` so the codec can reshape back
    to ``(NQ, T)`` for decoding.
    """
    results: list[Any] = []

    for src_idx in engine_input_source:
        if src_idx >= len(stage_list):
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        stage_out = stage_list[src_idx]
        audio_codes = _extract_audio_codes(stage_out)

        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2codec: no audio codes in stage output %d; emitting silence.", src_idx)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        # audio_codes: (T, NQ) → flatten to [NQ, T] → list[int]
        codes_nq_t = audio_codes.transpose(0, 1).contiguous()  # (NQ, T)
        flat = codes_nq_t.reshape(-1).tolist()

        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": codes_nq_t}},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Streaming (async chunk): called each time Stage 0 emits a new chunk
# ---------------------------------------------------------------------------


def talker2codec_delay_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit delay-patterned accumulated audio codes to Stage 1.

    State is maintained in ``transfer_manager`` keyed by request ID.
    A chunk is forwarded to Stage 1 when either:
      (a) ``is_finished`` is True (flush all remaining codes), or
      (b) the accumulated frame count reaches ``chunk_frames`` (default 25).

    Returns a dict compatible with the Stage-1 input format, or None to
    signal "not enough data yet — wait for more frames".
    """
    streaming = _moss_connector_extra(transfer_manager).get("moss_v1_streaming", False)
    if not isinstance(streaming, bool):
        raise ValueError(f"moss_v1_streaming must be a boolean, got {streaming!r}")
    if streaming:
        return talker2codec_delay_streaming_async_chunk(transfer_manager, multimodal_output, request, is_finished)

    req_id: str = str(getattr(request, "request_id", id(request)))
    pooling_output = multimodal_output

    # Initialise per-request accumulation state
    if not hasattr(transfer_manager, "_moss_tts_state"):
        transfer_manager._moss_tts_state = {}
    state = transfer_manager._moss_tts_state

    if req_id not in state:
        state[req_id] = {
            "accumulated": None,  # (T_acc, NQ) tensor or None
            "total_emitted": 0,
        }
    req_state = state[req_id]

    # Extract new codes from this chunk. The talker emits the full per-request
    # ``audio_codes["accumulated"]`` snapshot every step, so we only append the
    # *new* tail rows (otherwise we'd duplicate history quadratically).
    # ``pooling_output`` carries the unflattened OmniPayload at the top level
    # (``codes.audio``), matching the qwen3_tts pattern.
    if pooling_output is not None:
        codes_dict = pooling_output.get("codes", {}) or {}
        snapshot = codes_dict.get("audio")
        if isinstance(snapshot, torch.Tensor) and snapshot.numel() > 0:
            snapshot_cpu = snapshot.cpu()
            prev_t = 0 if req_state["accumulated"] is None else int(req_state["accumulated"].shape[0])
            new_rows = snapshot_cpu[prev_t:]
            if new_rows.numel() > 0:
                if req_state["accumulated"] is None:
                    req_state["accumulated"] = new_rows
                else:
                    req_state["accumulated"] = torch.cat([req_state["accumulated"], new_rows], dim=0)

    acc = req_state["accumulated"]
    if acc is None or acc.numel() == 0:
        if is_finished:
            del state[req_id]
        return None

    # The MOSS audio tokenizer's causal decoder doesn't yet have left-context
    # plumbing in this port, and a streaming chunk of 25 frames trips an
    # internal patched-pretransform reshape on the first chunk. Until we wire
    # left-context properly, accumulate all codes and emit only on finish.
    chunk_frames: int = 1 << 30
    left_context: int = 0

    t_acc = int(acc.shape[0])
    should_emit = is_finished or (t_acc - req_state["total_emitted"] >= chunk_frames)

    if not should_emit:
        return None

    # Determine the slice to emit
    emit_start = max(0, req_state["total_emitted"] - left_context)
    chunk_codes = acc[emit_start:]  # (T_chunk, NQ)
    req_state["total_emitted"] = t_acc

    if is_finished:
        del state[req_id]

    # Mirror upstream ``MossTTSDelayProcessor._parse_audio_codes``: the delay
    # talker samples codes in a delay pattern (a row is emitted every step, but
    # only the slot ``i == arange < audio_lengths`` carries a real code; the
    # rest is ``audio_pad_code``). Before sending to the codec we must
    #   1. de-delay   ``(T+nq-1, nq)`` → ``(T, nq)``
    #   2. drop rows that are entirely pad (separators between audio segments,
    #      and the leading text-mode rows that precede the first audio_start).
    chunk_codes_long = chunk_codes.to(torch.long).cpu().contiguous()  # (T_chunk, NQ)
    nq = int(chunk_codes_long.shape[1])
    t_chunk = int(chunk_codes_long.shape[0])
    audio_pad_code = 1024  # MOSS-TTS audio_pad_code; same value across variants.

    if t_chunk > nq:
        de_delayed = chunk_codes_long.new_zeros((t_chunk - nq + 1, nq))
        for i in range(nq):
            de_delayed[:, i] = chunk_codes_long[i : i + de_delayed.shape[0], i]
    else:
        de_delayed = chunk_codes_long.new_zeros((0, nq))

    if de_delayed.shape[0] > 0:
        is_pad = (de_delayed == audio_pad_code).all(dim=1)
        non_pad = ~is_pad
        if bool(non_pad.any()):
            de_delayed = de_delayed[non_pad]
        else:
            de_delayed = de_delayed.new_zeros((0, nq))

    if de_delayed.shape[0] == 0:
        # Nothing left after filtering — emit silence sentinel so the codec
        # request still completes cleanly.
        codec_flat: list[int] = []
    else:
        # Stage 1 (LLM_GENERATION codec) consumes ``codes.audio`` as a flat
        # codebook-major int list — chunk_transfer_adapter assigns it to
        # ``request.prompt_token_ids`` and the codec rebuilds the (NQ, T) grid.
        # Keep it a list (not a tensor): the receive path treats a list as the
        # token-id sequence directly, while a 1-D tensor would be ``.tolist()``-ed
        # anyway — the list keeps the downstream ``if not new_ids`` checks intact.
        codec_flat = de_delayed.transpose(0, 1).contiguous().reshape(-1).tolist()

    # main migrated the inter-stage connector from plain dicts to typed
    # ``OmniPayloadStruct`` (the chunk_transfer_adapter sender reads
    # ``payload_data.meta``); return the struct so MOSS matches that schema.
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_flat),
        meta=MetaStruct(
            left_context_size=left_context,
            finished=torch.tensor(bool(is_finished), dtype=torch.bool),
            # The chunk transfer adapter consumes ``finished`` for its own
            # bookkeeping and strips it from the Stage-1 runtime meta; the
            # codec-visible terminal flag must travel as ``stream_finished``
            # (same convention as the audex/cosyvoice3/qwen3_tts processors).
            stream_finished=torch.tensor(bool(is_finished), dtype=torch.bool),
        ),
    )


@dataclass
class _MossDelayStreamState:
    chunk_frames: int
    initial_chunk_frames: int
    seen_rows: int = 0
    nq: int | None = None
    delayed_tail: torch.Tensor | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    emitted: bool = False


def _moss_delay_chunk_sizes(transfer_manager: Any, request: Any) -> tuple[int, int]:
    config = _moss_connector_extra(transfer_manager)
    sizes = {
        "codec_chunk_frames": config.get("codec_chunk_frames", 8),
        "initial_codec_chunk_frames": config.get("initial_codec_chunk_frames", 8),
    }
    info = getattr(request, "additional_information", None)
    entries = getattr(info, "entries", {}) or {}
    for name in sizes:
        if name in entries:
            values = entries[name].list_data
            if values is None or len(values) != 1:
                raise ValueError(f"MOSS delay streaming {name} must contain one frame count")
            sizes[name] = values[0]
        value = sizes[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or int(value) != value
            or value <= 0
        ):
            raise ValueError(f"MOSS delay streaming {name} must be a positive integer, got {value!r}")
        sizes[name] = int(value)
    return sizes["codec_chunk_frames"], sizes["initial_codec_chunk_frames"]


def talker2codec_delay_streaming_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Incrementally de-delay cumulative MOSS v1 snapshots for a stateful codec.

    A frame at row t is complete only after delayed row t + NQ - 1 arrives.
    Keep that unfinished tail across calls and drop all-pad rows only after
    de-delay, matching full-sequence assembly. Chunk sizes default to 8/8 and
    can be overridden through connector config or request additional information.
    The terminal processor remains available as ``talker2codec_delay_async_chunk``.
    """
    external_id = getattr(request, "external_req_id", None)
    req_id = str(external_id if external_id is not None else getattr(request, "request_id", id(request)))
    if not hasattr(transfer_manager, "request_payload"):
        transfer_manager.request_payload = {}
    states = transfer_manager.request_payload
    state = states.get(req_id)
    if state is None:
        chunk_frames, initial_chunk_frames = _moss_delay_chunk_sizes(transfer_manager, request)
        state = _MossDelayStreamState(chunk_frames, initial_chunk_frames)
        # The transfer adapter clears this container on abort as well as finish.
        states[req_id] = state
    elif not isinstance(state, _MossDelayStreamState):
        raise ValueError(f"MOSS delay streaming found incompatible state for request {req_id}")

    if isinstance(multimodal_output, Mapping):
        codes = multimodal_output.get("codes", {}) or {}
        snapshot = codes.get("audio")
        if isinstance(snapshot, torch.Tensor) and snapshot.numel() > 0:
            if snapshot.ndim != 2 or snapshot.shape[1] <= 0:
                raise ValueError(f"MOSS delayed snapshot must be [T, NQ], got {tuple(snapshot.shape)}")
            rows, nq = map(int, snapshot.shape)
            if state.nq is not None and nq != state.nq:
                raise ValueError(f"MOSS delayed snapshot changed NQ from {state.nq} to {nq}")
            if rows < state.seen_rows:
                raise ValueError(f"MOSS cumulative snapshot shrank from {state.seen_rows} to {rows} rows")
            state.nq = nq
            if rows > state.seen_rows:
                # Slice before host transfer: the talker republishes all history.
                new_rows = snapshot[state.seen_rows :].detach().to(device="cpu", dtype=torch.long)
                delayed = new_rows if state.delayed_tail is None else torch.cat([state.delayed_tail, new_rows], dim=0)
                ready = max(0, int(delayed.shape[0]) - nq + 1)
                if ready:
                    frames = torch.stack([delayed[i : i + ready, i] for i in range(nq)], dim=1)
                    frames = frames[frames.ne(_MOSS_AUDIO_PAD_CODE).any(dim=1)]
                    state.pending.extend(frames.unbind(dim=0))
                # Clone so a short tail cannot retain a whole CPU snapshot's storage.
                state.delayed_tail = delayed[ready:].clone()
                state.seen_rows = rows

    threshold = state.chunk_frames if state.emitted else state.initial_chunk_frames
    if not is_finished and len(state.pending) < threshold:
        return None

    emit_frames = len(state.pending) if is_finished else threshold
    if emit_frames:
        chunk_codes = torch.stack(state.pending[:emit_frames], dim=0)
        codec_flat = chunk_codes.transpose(0, 1).contiguous().reshape(-1).tolist()
        del state.pending[:emit_frames]
    else:
        codec_flat = []
    state.emitted = True
    if is_finished:
        states.pop(req_id, None)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_flat),
        meta=MetaStruct(
            req_id=[req_id],
            left_context_size=0,
            codec_chunk_frames=emit_frames,
            codec_left_context_frames=0,
            code_flat_numel=len(codec_flat),
            codec_streaming=True,
            stream_finished=torch.tensor(bool(is_finished), dtype=torch.bool),
            finished=torch.tensor(bool(is_finished), dtype=torch.bool),
        ),
        request_id=req_id,
    )


def talker2codec_raw_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async processor for MOSS-TTS Local/Realtime raw codec rows.

    Stage 0 emits newly generated raw codec rows shaped ``[T, n_vq]`` (normally
    ``[1, n_vq]`` per decode step). This processor buffers those new rows until
    a codec chunk is ready, then forwards the chunk to Stage 1. No delay-pattern
    de-delay is applied on this path.
    """
    external_req_id = getattr(request, "external_req_id", None)
    req_id = str(external_req_id if external_req_id is not None else getattr(request, "request_id", id(request)))

    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)
    if not hasattr(transfer_manager, "request_payload"):
        transfer_manager.request_payload = {}
    if not hasattr(transfer_manager, "put_req_chunk"):
        transfer_manager.put_req_chunk = defaultdict(int)

    pending_frames = transfer_manager.code_prompt_token_ids[req_id]

    if isinstance(multimodal_output, Mapping):
        codes_dict = multimodal_output.get("codes", {}) or {}
        new_frames = codes_dict.get("audio")
        if isinstance(new_frames, torch.Tensor) and new_frames.numel() > 0:
            frames_cpu = new_frames.detach().to("cpu", torch.long).contiguous()
            if frames_cpu.ndim == 1:
                frames_cpu = frames_cpu.reshape(1, -1)
            if frames_cpu.ndim != 2:
                raise ValueError(f"MOSS raw codec frames must be 2-D, got {tuple(frames_cpu.shape)}")
            valid_rows = frames_cpu.ne(_MOSS_AUDIO_PAD_CODE).any(dim=1)
            for frame in frames_cpu[valid_rows]:
                pending_frames.append(frame.clone())
        # Raw/local streaming should mirror the non-streaming path: the codec
        # decodes only generated audio rows. Reference audio conditions the
        # talker, but feeding its codes into the codec streaming state adds a
        # long first-packet prime step and changes the decoder state relative
        # to non-streaming output.

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    chunk_frames = int(cfg.get("codec_chunk_frames", 15) or 15)
    initial_chunk_frames = int(cfg.get("initial_codec_chunk_frames") or 0)
    if chunk_frames <= 0:
        raise ValueError(f"codec_chunk_frames must be positive for MOSS raw streaming, got {chunk_frames}")
    if initial_chunk_frames < 0:
        raise ValueError(f"initial_codec_chunk_frames must be non-negative, got {initial_chunk_frames}")
    if initial_chunk_frames > chunk_frames:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping.",
            initial_chunk_frames,
            chunk_frames,
        )
        initial_chunk_frames = chunk_frames

    pending = len(pending_frames)
    emitted_any = int(transfer_manager.put_req_chunk.get(req_id, 0)) > 0
    threshold = initial_chunk_frames if initial_chunk_frames > 0 and not emitted_any else chunk_frames
    if pending <= 0:
        if is_finished:
            transfer_manager.code_prompt_token_ids.pop(req_id, None)
            transfer_manager.request_payload.pop(req_id, None)
            return OmniPayloadStruct(
                meta=MetaStruct(
                    req_id=[req_id],
                    left_context_size=0,
                    codec_chunk_frames=0,
                    codec_left_context_frames=0,
                    stream_finished=torch.tensor(True, dtype=torch.bool),
                    finished=torch.tensor(True, dtype=torch.bool),
                ),
                request_id=req_id,
            )
        return None
    if not is_finished and pending < threshold:
        return None

    emit_frames = pending if is_finished else threshold
    chunk_rows = pending_frames[:emit_frames]
    del pending_frames[:emit_frames]
    chunk_codes = torch.stack(
        [row.to(torch.long).cpu() for row in chunk_rows],
        dim=0,
    ).contiguous()
    finished = bool(is_finished and len(pending_frames) == 0)

    codec_flat = chunk_codes.transpose(0, 1).contiguous().reshape(-1).to(torch.long)

    if finished:
        transfer_manager.code_prompt_token_ids.pop(req_id, None)
        transfer_manager.request_payload.pop(req_id, None)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_flat),
        meta=MetaStruct(
            req_id=[req_id],
            left_context_size=0,
            codec_chunk_frames=int(chunk_codes.shape[0]),
            codec_left_context_frames=0,
            code_flat_numel=int(codec_flat.numel()),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
        request_id=req_id,
    )


__all__ = [
    "talker2codec",
    "talker2codec_delay_async_chunk",
    "talker2codec_delay_streaming_async_chunk",
    "talker2codec_raw_async_chunk",
]
