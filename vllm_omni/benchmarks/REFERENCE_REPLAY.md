# Reference-token replay

`reference_replay_worker.py` installs an instance-level V1 `_sample` wrapper through vLLM's worker-extension RPC interface.
The wrapper calls the original sampler after every model forward, then replaces the sampled token IDs before the runner updates request state.
It does not edit the installed vLLM package or register a logits processor.

This is a latency experiment utility, not a model-quality benchmark.
Following a reference answer changes the generated trajectory and does not establish the latency of free generation.

## Tested scope

The local decoder test uses vLLM 0.26.0, Qwen3-0.6B, the V1 runner, asynchronous scheduling, CUDA graphs, and one GPU.
Prefix caching is disabled for every arm to measure the same prefill work.
Omni's worker selects V1, but this small-model test does not run the full Omni speech pipeline.
The repository targets a different vLLM version, so importing `vllm_omni` and enabling this extension in a full Omni worker require separate compatibility checks.

Use the standalone module import used by the test:

```bash
# From the vllm-omni repository; select only a GPU assigned to your task.
export PYTHONPATH="$PWD/vllm_omni/benchmarks:$PYTHONPATH"
export VLLM_USE_V2_MODEL_RUNNER=0
```

The extension rejects speculative decoding, hybrid models, multiple samples per request, and requested output or prompt log probabilities.
It supports one sampled token per row and uses the V1 position buffers.
Reference IDs must come from the exact model tokenizer and include the intended EOS or other terminating token.
The current binding requires `max_tokens == len(reference_ids)`; natural early EOS with a larger original cap needs a separate, explicitly defined comparison.
The decoder latency cases use natural trajectories that reach their declared caps.

## Usage

Create the engine with the standalone class name:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    worker_extension_cls="reference_replay_worker.ReferenceReplayWorkerExtension",
    enable_prefix_caching=False,
)
engine = llm.llm_engine

# IDs are prepared before timing. Capture natural IDs with this same engine
# for a paired latency comparison, or tokenize an explicit reference response.
prompt_ids = [100, 200, 300]
reference_ids = [400, 500]
references = {
    "example": {
        "prompt_token_ids": prompt_ids,
        "token_ids": reference_ids,
    }
}
params = SamplingParams(
    temperature=0,
    max_tokens=len(reference_ids),
    extra_args={"reference_replay_key": "example"},
)

assert not engine.has_unfinished_requests()
engine.collective_rpc(
    "reference_replay_configure", args=("replay", references)
)
outputs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
assert not engine.has_unfinished_requests()
audit = engine.collective_rpc("reference_replay_finish")
assert outputs[0].outputs[0].token_ids == reference_ids

# Restore the original method, with no wrapper bypass on the native path.
engine.collective_rpc("reference_replay_configure", args=("native",))
```

`noop` uses the same reference binding and view selection but keeps the original sampled IDs.
`replay` returns the reference IDs.
`native` restores the original bound method, including removal of the instance override when the method was inherited.
Run `reference_replay_finish` before changing modes after any hooked cohort.
All configure and finish calls require an idle engine; the caller must enforce this contract.
After a callback failure or an invalid offset, retain the failure and shut down the engine instead of clearing the audit.

Requests bind through `SamplingParams.extra_args["reference_replay_key"]` and the runner's actual internal request ID.
The hook checks the exact prompt and forbids reusing a reference key for a different request in one registration.
Batch additions, removals, and row moves rebuild the current mapping inside the sample call.
Requests without a replay key keep their original sampled IDs.

## Offset and output ownership

For non-speculative V1, the CPU and GPU paths use the same computed-token and scheduled-token counts.
The runner copies computed-token counts unchanged to GPU, then derives GPU positions and sequence lengths from those counts.
In the tested 0.26 source, these steps are in `gpu_model_runner.py` at lines 2062–2066, 2130–2155, and 2163–2171.
Subtracting each prompt length from `optimistic_seq_lens_cpu` therefore gives the current output offset without copying token history from GPU.
When every active row has the same valid offset, the hook selects a preloaded time-major GPU view.
Noop selects that view too; it does not run a per-step GPU gather on this path.
The controller retains CPU memory views and pre-created GPU frame views instead of converting the same CPU tensor and constructing a GPU slice each step.
It reads the current sequence and prompt lengths on every call, and refreshes a cached view if its source object changes.
The views retain their backing storage; they do not replace actual positions with a call counter.
Mixed offsets, untracked rows, partial prefill, and boundary samples use the fused GPU path.
The first mapping or row-reordering operation remains inside the sample call.
The full canonical slot vector is preloaded during the idle configuration RPC, so the first canonical binding does not copy it from CPU to GPU.
Noncanonical bindings still create a CUDA index tensor from CPU and may synchronize; dynamic mappings are supported, but their copy cost is not eliminated by this change.

The 0.26 bookkeeping and asynchronous output paths retain or copy the sampled IDs without modifying that GPU view in place.
This ownership assumption must be checked when upgrading vLLM.
Partial-prefill samples and one async sample exactly at the `max_tokens` boundary remain unchanged and receive separate audit counters.
An offset past that boundary fails the final audit.
The caller must still compare every emitted ID and termination condition; the counters alone do not prove output correctness.

## Tests

The focused test calls the installed GPU `Sampler` through the actual wrapper and uses small request/position fixtures.
It checks row movement, additions/removals, mixed offsets, partial prefill, boundary errors, exception retention, and native restoration.
It also checks in-place length updates, replacement of both CPU source arrays, and the lifetime of GPU frames after a row reorder.
It does not load an Omni model.

```bash
CUDA_VISIBLE_DEVICES=1 python tests/benchmarks/test_reference_replay_worker.py
```

Use a GPU allocated to your task and the CUDA libraries required by your local vLLM installation.
The experiment runner also records real decoder trajectories, EOS behavior, per-request timing, and source snapshots.
