# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only post-sampler replay via vLLM's worker-extension RPC API.

The frontend must call configure/finish only while its engine is idle. No
logits processor is registered, so the original async history requirements
and sampler remain unchanged. This implementation targets the V1 runner.
"""

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _replace_sample(
    sampled,
    references,
    lengths,
    prompt_lengths,
    row_slots,
    positions,
    query_start_loc,
    counters,
    sample_stride: tl.constexpr,
    reference_width: tl.constexpr,
    num_rows: tl.constexpr,
    FORCE: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    rows = tl.arange(0, BLOCK)
    active = rows < num_rows
    slots = tl.load(row_slots + rows, mask=active, other=-1)
    tracked = active & (slots >= 0)
    last_input = tl.load(query_start_loc + rows + 1, mask=tracked, other=1) - 1
    absolute_position = tl.load(positions + last_input, mask=tracked, other=0)
    prompt_len = tl.load(prompt_lengths + slots, mask=tracked, other=1)
    offset = absolute_position - prompt_len + 1
    length = tl.load(lengths + slots, mask=tracked, other=0)
    valid = tracked & (offset >= 0) & (offset < length)
    target = tl.load(references + slots * reference_width + offset, mask=valid, other=0)
    original = tl.load(sampled + rows * sample_stride, mask=active, other=0)
    value = tl.where(valid & FORCE, target, original)
    tl.store(sampled + rows * sample_stride, value, mask=active)
    tl.atomic_add(counters + 0, tl.sum(valid.to(tl.int32)))
    tl.atomic_add(counters + 1, tl.sum((tracked & (offset < 0)).to(tl.int32)))
    # V1 can speculatively execute one discarded step at the max_tokens bound.
    tl.atomic_add(counters + 2, tl.sum((tracked & (offset == length)).to(tl.int32)))
    tl.atomic_add(counters + 3, tl.sum((tracked & (offset > length)).to(tl.int32)))
    # Keep the reference gather observable in noop as well as replay.
    tl.atomic_add(counters + 4, tl.sum(tl.where(valid, target, 0).to(tl.int64)))


class _ReplayController:
    def __init__(self, runner, mode, references):
        if mode not in ("noop", "replay"):
            raise ValueError("mode must be noop or replay")
        if runner.vllm_config.speculative_config is not None:
            raise ValueError("post-sampler replay does not support speculative decoding")
        if runner.model_config.is_hybrid:
            raise ValueError("post-sampler replay does not support hybrid models")
        if not torch.is_tensor(runner.positions) or not hasattr(runner.query_start_loc, "gpu"):
            raise ValueError("post-sampler replay requires the V1 positions/query buffers")
        if not isinstance(references, dict) or not 1 <= len(references) <= runner.max_num_reqs:
            raise ValueError("invalid bounded reference inventory")
        self.mode = mode
        self.references = references
        self.keys = list(references)
        self.slots = {key: index for index, key in enumerate(self.keys)}
        vocab_size = runner.model_config.get_vocab_size()
        for key, row in references.items():
            if not isinstance(key, str) or set(row) != {"prompt_token_ids", "token_ids"}:
                raise ValueError("each reference requires a string key, prompt_token_ids and token_ids")
            if not isinstance(row["token_ids"], list) or not 1 <= len(row["token_ids"]) <= 65536:
                raise ValueError("invalid reference length")
            if not isinstance(row["prompt_token_ids"], list) or not row["prompt_token_ids"]:
                raise ValueError("prompt IDs are required for exact request binding")
            if any(
                type(token) is not int or not 0 <= token < vocab_size
                for token in row["token_ids"] + row["prompt_token_ids"]
            ):
                raise ValueError("invalid token ID in reference inventory")
        self.width = max(len(row["token_ids"]) for row in references.values())
        table = [row["token_ids"] + [0] * (self.width - len(row["token_ids"])) for row in references.values()]
        self.table = torch.tensor(table, device=runner.device, dtype=torch.long)
        self.time_major = self.table.transpose(0, 1).contiguous().to(torch.int32).unsqueeze(-1)
        self.lengths = torch.tensor(
            [len(row["token_ids"]) for row in references.values()], device=runner.device, dtype=torch.long
        )
        self.prompt_lengths = torch.tensor(
            [len(row["prompt_token_ids"]) for row in references.values()], device=runner.device, dtype=torch.long
        )
        self.counters = torch.zeros(5, device=runner.device, dtype=torch.int64)
        self.canonical_row_slots = torch.arange(len(self.keys), device=runner.device, dtype=torch.long)
        self.batch_ids = None
        self.row_slots = None
        self.bindings = []
        self.owners = {}
        self.sample_calls = 0
        self.checked = False
        self.failure = None
        self.fast_calls = 0
        self.fast_valid = 0
        self.fast_checksum = 0
        self.batch_view = None

    def bind(self, runner, ids):
        slots = []
        for rid in ids:
            request = runner.requests[rid]
            params = request.sampling_params
            key = (params.extra_args or {}).get("reference_replay_key")
            if key is None:
                slots.append(-1)
                continue
            if key not in self.references:
                raise ValueError("request references an unregistered replay key")
            reference = self.references[key]
            if request.prompt_token_ids != reference["prompt_token_ids"]:
                raise ValueError("request prompt differs from registered reference")
            if params.max_tokens != len(reference["token_ids"]) or params.n != 1:
                raise ValueError("reference length and request max_tokens/n differ")
            if params.logprobs is not None or params.prompt_logprobs is not None:
                raise ValueError("post-sample replay does not expose sampled-distribution logprobs")
            if key in self.owners and self.owners[key] != rid:
                raise ValueError("replay key was reused by another request")
            self.owners[key] = rid
            slots.append(self.slots[key])
        self.batch_ids = ids
        self.row_slots = (
            self.canonical_row_slots
            if slots == list(range(len(self.keys)))
            else torch.tensor(slots, device=runner.device, dtype=torch.long)
        )
        self.bindings.append({"request_ids": list(ids), "reference_slots": slots})
        self.batch_view = None
        if slots and all(slot >= 0 for slot in slots):
            self.batch_view = (
                self.time_major
                if slots == list(range(len(self.keys)))
                else self.time_major.index_select(1, self.row_slots)
            )
            rows = [self.references[self.keys[slot]]["token_ids"] for slot in slots]
            self.common_length = min(map(len, rows))
            self.batch_checksums = [sum(row[index] for row in rows) for index in range(self.common_length)]

    def apply(self, runner, sampled, spec_decode_metadata):
        self.checked = False
        if (
            spec_decode_metadata is not None
            or sampled.sampled_token_ids.ndim != 2
            or sampled.sampled_token_ids.shape[1] != 1
        ):
            raise ValueError("post-sample replay requires one non-speculative token per row")
        ids = tuple(runner.input_batch.req_ids)
        if ids != self.batch_ids:
            self.bind(runner, ids)
        if not ids:
            return sampled
        if len(ids) != sampled.sampled_token_ids.shape[0]:
            raise ValueError("sampler rows differ from request inventory")
        self.sample_calls += 1
        if self.batch_view is not None and sampled.sampled_token_ids.dtype == self.batch_view.dtype:
            # Without speculative decoding, V1 copies these CPU lengths to GPU
            # unchanged before computing positions (gpu_model_runner.py:2152).
            offsets = (
                runner.optimistic_seq_lens_cpu.numpy()[: len(ids)] - runner.input_batch.num_prompt_tokens[: len(ids)]
            )
            offset = int(offsets[0])
            if 0 <= offset < self.common_length and (offsets == offset).all():
                reference_view = self.batch_view[offset]
                self.fast_calls += 1
                self.fast_valid += len(ids)
                self.fast_checksum += self.batch_checksums[offset]
                if self.mode == "replay":
                    sampled.sampled_token_ids = reference_view
                return sampled
        _replace_sample[(1,)](
            sampled.sampled_token_ids,
            self.table,
            self.lengths,
            self.prompt_lengths,
            self.row_slots,
            runner.positions,
            runner.query_start_loc.gpu,
            self.counters,
            sampled.sampled_token_ids.stride(0),
            self.width,
            len(ids),
            self.mode == "replay",
            triton.next_power_of_2(len(ids)),
        )
        return sampled

    def finish(self):
        if self.failure is not None:
            raise RuntimeError(f"replay callback previously failed: {self.failure}")
        counts = self.counters.cpu().tolist()
        counts[0] += self.fast_valid
        counts[4] += self.fast_checksum
        result = {
            "mode": self.mode,
            "sample_calls": self.sample_calls,
            "valid_reference_lookups": counts[0],
            "partial_prefill_samples_left_unchanged": counts[1],
            "max_tokens_tail_samples_left_unchanged": counts[2],
            "out_of_range": counts[3],
            "reference_token_checksum": counts[4],
            "common_offset_view_calls": self.fast_calls,
            "fused_calls": self.sample_calls - self.fast_calls,
            "batch_bindings": self.bindings,
            "unused_reference_keys": sorted(set(self.keys) - set(self.owners)),
        }
        if counts[3]:
            raise RuntimeError(f"reference offset exceeded the declared request boundary: {result}")
        self.checked = True
        return result


def _sample_with_reference(self, logits, spec_decode_metadata):
    controller = self._reference_replay_controller
    controller.checked = False
    try:
        sampled = self._reference_replay_original(logits, spec_decode_metadata)
        return controller.apply(self, sampled, spec_decode_metadata)
    except BaseException as exc:
        controller.failure = f"{type(exc).__name__}: {exc}"
        raise


class ReferenceReplayWorkerExtension:
    def reference_replay_configure(self, mode, references=None):
        """Idle-only RPC: preload references and install/remove the instance hook."""
        runner = self.model_runner
        old = getattr(runner, "_reference_replay_controller", None)
        if old is not None and not old.checked:
            raise RuntimeError("finish the previous replay audit before reconfiguration")
        if hasattr(runner, "_reference_replay_original"):
            if runner._reference_replay_had_override:
                runner._sample = runner._reference_replay_original
            else:
                del runner._sample
            del runner._reference_replay_controller
            del runner._reference_replay_original
            del runner._reference_replay_had_override
        if mode == "native":
            return {"mode": "native", "instance_hook_installed": False}
        controller = _ReplayController(runner, mode, references)
        runner._reference_replay_had_override = "_sample" in runner.__dict__
        runner._reference_replay_original = runner._sample
        runner._reference_replay_controller = controller
        runner._sample = MethodType(_sample_with_reference, runner)
        return {"mode": mode, "instance_hook_installed": True, "reference_count": len(references)}

    def reference_replay_finish(self):
        """Idle-only RPC: check GPU boundary counters after timed output consumption."""
        controller = getattr(self.model_runner, "_reference_replay_controller", None)
        return controller.finish() if controller is not None else {"mode": "native", "instance_hook_installed": False}
