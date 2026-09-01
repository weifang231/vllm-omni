# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for trusted Qwen3-Omni chat playback-start buffering."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponseStreamChoice
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.outputs import CompletionOutput, RequestOutput

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
)
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.openai.playback_start import PlaybackStartConfig
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_audio_output(
    marker: str,
    *,
    audio_samples: int = 100,
    sample_rate: int = 1000,
    finish_reason: str | None = None,
) -> OmniRequestOutput:
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason=finish_reason,
        stop_reason=None,
    )
    completion.multimodal_output = {
        "audio": [MagicMock(numel=MagicMock(return_value=audio_samples))],
        "marker": marker,
        "sr": sample_rate,
    }
    output = RequestOutput(
        request_id="test-req",
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[completion],
        finished=finish_reason is not None,
    )
    return OmniRequestOutput.from_stage_output(
        output,
        request_id="test-req",
        stage_id=2,
        replica_id=7,
        final_output_type="audio",
        finished=finish_reason is not None,
    )


def _install_marker_audio_serializer(serving_chat) -> None:
    def create_audio_choice(omni_res, role, request, stream=False):
        del request, stream
        output = omni_res.outputs[0]
        return [
            ChatCompletionResponseStreamChoice(
                index=output.index,
                delta=DeltaMessage(
                    role=role,
                    content=output.multimodal_output["marker"],
                ),
                logprobs=None,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )
        ]

    serving_chat._create_audio_choice = create_audio_choice


def _raw_request(headers: dict[str, str] | None = None):
    return SimpleNamespace(headers=headers or {}, state=SimpleNamespace())


def _payload(line: str) -> dict:
    assert line.startswith("data: ")
    return json.loads(line.removeprefix("data: ").strip())


@pytest.mark.asyncio
async def test_chat_admission_and_playback_share_ingress_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_omni.entrypoints.openai.serving_chat as serving_chat_module

    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    monkeypatch.setattr(
        serving_chat_module,
        "time",
        SimpleNamespace(monotonic=lambda: 10.0, time=time.time),
    )
    engine_client = MagicMock()
    engine_client.errored = False
    engine_client.output_modalities = ["text", "audio"]
    engine_client.stage_configs = []
    engine_client.renderer = SimpleNamespace(get_tokenizer=lambda: object())
    engine_client.generate = MagicMock(return_value=object())
    serving_chat = build_serving_chat(
        engine_client=engine_client,
        models=SimpleNamespace(model_name=lambda _adapter: "test-model"),
        online_renderer=SimpleNamespace(validate_chat_template=lambda **_kwargs: None),
    )
    serving_chat.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_omni_moe"))
    serving_chat._check_model = AsyncMock(return_value=None)
    serving_chat._maybe_get_adapters = MagicMock(return_value=None)
    serving_chat._effective_chat_template_kwargs = MagicMock(return_value={})
    serving_chat._preprocess_chat = AsyncMock(return_value=([], [{"prompt": "processed"}]))
    serving_chat._base_request_id = MagicMock(return_value="test")
    serving_chat._build_sampling_params_list_from_request = MagicMock(return_value=[MagicMock()])
    serving_chat._log_inputs = MagicMock()
    serving_chat.chat_completion_stream_generator = MagicMock(return_value="done")
    raw_request = SimpleNamespace(
        headers={
            "x-vllm-omni-playback-buffer-ms": "300",
            "x-vllm-omni-first-output-deadline-ms": "900",
        },
        state=SimpleNamespace(request_timestamp=time.time()),
    )

    result = await serving_chat._create_chat_completion(
        make_request(modalities=["text", "audio"]),
        raw_request,
    )

    assert result == "done"
    scheduling_deadline = engine_client.generate.call_args.kwargs["first_output_deadline_monotonic_s"]
    playback_config = serving_chat.chat_completion_stream_generator.call_args.kwargs["playback_start_config"]
    assert scheduling_deadline == 10.9
    assert playback_config.deadline_monotonic_s == scheduling_deadline


def _stream(serving_chat, request, result_generator, raw_request, config):
    return serving_chat.chat_completion_stream_generator(
        request=request,
        result_generator=result_generator,
        request_id="test-req",
        model_name="test-model",
        conversation=[],
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
        raw_request=raw_request,
        playback_start_config=config,
    )


def test_chat_playback_header_is_trusted_opt_in_for_qwen3_omni(monkeypatch: pytest.MonkeyPatch) -> None:
    serving_chat = build_serving_chat()
    serving_chat.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_omni_moe"))
    request = make_request(modalities=["text", "audio"])
    raw_request = _raw_request(
        {
            "x-vllm-omni-playback-buffer-ms": "250",
            "x-vllm-omni-first-output-deadline-ms": "900",
        }
    )

    monkeypatch.delenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", raising=False)
    assert (
        serving_chat._chat_playback_start_config(
            request,
            raw_request,
            request_start_s=10.0,
        )
        is None
    )

    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    config = serving_chat._chat_playback_start_config(
        request,
        raw_request,
        request_start_s=10.0,
    )
    assert config is not None
    assert config.target_ms == 250.0
    assert config.deadline_monotonic_s == pytest.approx(10.9)

    assert (
        serving_chat._chat_playback_start_config(
            make_request(modalities=["text"]),
            raw_request,
            request_start_s=10.0,
        )
        is None
    )
    assert (
        serving_chat._chat_playback_start_config(
            make_request(modalities=["audio"], stream=False),
            raw_request,
            request_start_s=10.0,
        )
        is None
    )

    serving_chat.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    assert (
        serving_chat._chat_playback_start_config(
            request,
            raw_request,
            request_start_s=10.0,
        )
        is None
    )


@pytest.mark.asyncio
async def test_chat_holds_only_audio_drains_engine_and_preserves_audio_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.entrypoints.openai.serving_chat as serving_chat_mod

    serving_chat = build_serving_chat()
    _install_marker_audio_serializer(serving_chat)
    request = make_request(modalities=["text", "audio"])
    raw_request = _raw_request()
    req_state = ClientRequestState(
        request_id="internal-req",
        external_request_id="test-req",
    )
    req_state.request_arrival_ts = time.time() - 1.0
    serving_chat.engine_client.request_states = {"internal-req": req_state}
    serving_chat.engine_client.mod_metrics = object()
    monkeypatch.setattr(serving_chat_mod, "observe_audio_first_packet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serving_chat_mod, "observe_audio_streaming_finalize", lambda *_args, **_kwargs: None)

    engine_waiting = asyncio.Event()
    allow_second_audio = asyncio.Event()

    async def results():
        yield _make_audio_output("first")
        engine_waiting.set()
        yield make_text_omni_output(text="hello", token_ids=[10], finish_reason="stop")
        await allow_second_audio.wait()
        yield _make_audio_output("second", finish_reason="stop")

    stream = _stream(
        serving_chat,
        request,
        results(),
        raw_request,
        PlaybackStartConfig(target_ms=150.0),
    )

    first_text = _payload(await asyncio.wait_for(anext(stream), timeout=1.0))
    second_text = _payload(await asyncio.wait_for(anext(stream), timeout=1.0))
    assert engine_waiting.is_set()
    assert first_text["modality"] == second_text["modality"] == "text"
    assert req_state.first_audio_ts is None
    assert req_state.audio_chunk_arrivals_s == []

    pending_audio = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert not pending_audio.done()
    allow_second_audio.set()

    first_audio = _payload(await asyncio.wait_for(pending_audio, timeout=1.0))
    second_audio = _payload(await asyncio.wait_for(anext(stream), timeout=1.0))
    assert [
        first_audio["choices"][0]["delta"]["content"],
        second_audio["choices"][0]["delta"]["content"],
    ] == ["first", "second"]
    assert req_state.first_audio_ts is not None
    assert req_state.audio_chunk_bytes == [200, 200]
    assert req_state.audio_sample_rate == 1000

    remaining = await collect_stream(stream)
    assert remaining[-1] == "data: [DONE]\n\n"
    telemetry = raw_request.state.playback_start_telemetry
    assert telemetry["status"] == "ok"
    assert telemetry["release_reason"] == "target"
    assert telemetry["buffered_audio_ms"] == 200.0


@pytest.mark.asyncio
async def test_chat_deadline_flushes_audio_without_cancelling_engine_pull() -> None:
    serving_chat = build_serving_chat()
    _install_marker_audio_serializer(serving_chat)
    request = make_request(modalities=["audio"])
    raw_request = _raw_request()
    engine_waiting = asyncio.Event()
    allow_second_audio = asyncio.Event()

    async def results():
        yield _make_audio_output("first")
        engine_waiting.set()
        await allow_second_audio.wait()
        yield _make_audio_output("second", finish_reason="stop")

    stream = _stream(
        serving_chat,
        request,
        results(),
        raw_request,
        PlaybackStartConfig(
            target_ms=500.0,
            deadline_monotonic_s=time.monotonic() + 0.01,
        ),
    )
    first_audio = _payload(await asyncio.wait_for(anext(stream), timeout=1.0))
    assert first_audio["choices"][0]["delta"]["content"] == "first"
    assert engine_waiting.is_set()

    allow_second_audio.set()
    second_audio = _payload(await asyncio.wait_for(anext(stream), timeout=1.0))
    assert second_audio["choices"][0]["delta"]["content"] == "second"
    await collect_stream(stream)

    telemetry = raw_request.state.playback_start_telemetry
    assert telemetry["status"] == "ok"
    assert telemetry["release_reason"] == "deadline"
    assert telemetry["deadline_fallback"] is True
    assert telemetry["buffered_audio_ms"] == 100.0
    assert telemetry["hold_ms"] > 0.0


@pytest.mark.asyncio
async def test_chat_eos_flushes_short_audio() -> None:
    serving_chat = build_serving_chat()
    _install_marker_audio_serializer(serving_chat)
    request = make_request(modalities=["audio"])
    raw_request = _raw_request()

    async def results():
        yield _make_audio_output("short", finish_reason="stop")

    lines = await collect_stream(
        _stream(
            serving_chat,
            request,
            results(),
            raw_request,
            PlaybackStartConfig(target_ms=500.0),
        )
    )
    payloads = [_payload(line) for line in lines if line != "data: [DONE]\n\n"]
    audio_payloads = [payload for payload in payloads if payload.get("modality") == "audio"]
    assert [payload["choices"][0]["delta"]["content"] for payload in audio_payloads] == ["short"]
    assert raw_request.state.playback_start_telemetry["release_reason"] == "eos"


@pytest.mark.asyncio
async def test_chat_error_discards_held_audio() -> None:
    serving_chat = build_serving_chat()
    _install_marker_audio_serializer(serving_chat)
    request = make_request(modalities=["audio"])
    raw_request = _raw_request()

    async def results():
        yield _make_audio_output("held")
        raise RuntimeError("boom")

    lines = await collect_stream(
        _stream(
            serving_chat,
            request,
            results(),
            raw_request,
            PlaybackStartConfig(target_ms=500.0),
        )
    )
    payloads = [_payload(line) for line in lines if line != "data: [DONE]\n\n"]
    assert all(payload.get("modality") != "audio" for payload in payloads)
    telemetry = raw_request.state.playback_start_telemetry
    assert telemetry["status"] == "error"
    assert telemetry["release_reason"] == "error"


@pytest.mark.asyncio
async def test_chat_cancellation_discards_held_audio_and_closes_engine() -> None:
    serving_chat = build_serving_chat()
    _install_marker_audio_serializer(serving_chat)
    request = make_request(modalities=["audio"])
    raw_request = _raw_request()
    engine_waiting = asyncio.Event()
    engine_closed = asyncio.Event()

    async def results():
        try:
            yield _make_audio_output("held")
            engine_waiting.set()
            await asyncio.Event().wait()
        finally:
            engine_closed.set()

    stream = _stream(
        serving_chat,
        request,
        results(),
        raw_request,
        PlaybackStartConfig(
            target_ms=500.0,
            deadline_monotonic_s=time.monotonic() + 60.0,
        ),
    )
    delivery = asyncio.create_task(anext(stream))
    await asyncio.wait_for(engine_waiting.wait(), timeout=1.0)
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery

    assert engine_closed.is_set()
    telemetry = raw_request.state.playback_start_telemetry
    assert telemetry["status"] == "cancelled"
    assert telemetry["release_reason"] == "cancelled"
