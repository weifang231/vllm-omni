# Runtime Queue Control

## Development baseline

- Fork remote: `origin = https://github.com/weifang231/vllm-omni.git`
- Official remote: `upstream = https://github.com/vllm-project/vllm-omni.git`
- Base commit: `a6b559cf54ff1c3c8c0a386f3c93fcd8ff1bde41`
- Base branch: `main`
- Feature branch: `feature/runtime-queue-control`
- Recorded: `2026-08-30`

The official remote has a disabled push URL. Development is based on the
common `origin/main` and `upstream/main` commit above.

## Scope

This feature adds an optional coordinator-owned queue-control layer for
multi-stage request dispatch. It is disabled by default, so existing request
submission, batching, placement, cache management, and GPU execution remain
unchanged unless an operator explicitly supplies a queue-control policy.

The primary contract is the paper's class-level, end-to-end concurrency lease:
a request acquires global, path, and class credit immediately before its first
stage-0 submission and holds that lease until success, cancellation, or failure.
Per-stage WIP limits are a separately configured runtime extension; they are
not an implementation of the paper's current class-level optimizer.

The control layer is responsible only for:

1. enforcing configured pipeline, path, class, and per-stage work-in-progress
   limits without preempting requests that already hold a lease;
2. ordering queued stage submissions by FIFO or absolute first-output deadline;
3. carrying observable request class, path, and deadline metadata from the
   public engine API through the orchestrator and downstream stage requests;
4. exposing bounded-cardinality queue, active-lease, dispatch, and wait-time
   telemetry for evaluation and controller feedback.

It does not implement the paper's admission score, dynamic-program optimizer,
or playback-start rule. Those policies may update this runtime mechanism, but
must not be represented as implemented merely because the mechanism exists.

Full-duplex session submissions currently preserve scheduling metadata but do
not pass through this queue. The standard Qwen3-Omni and Qwen3-TTS request,
streaming-update, CFG-companion, inter-stage-forward, and async-chunk-prewarm
paths do.

## Control file

Set `VLLM_OMNI_RUNTIME_CONTROL_FILE` to a JSON file that is replaced atomically
by the controller. The orchestrator polls it without preempting active work:

```json
{
  "queue_control": {
    "enabled": true,
    "policy": "edf",
    "global_wip_limit": 8,
    "path_wip_limits": {"audio": 7, "text": 1},
    "class_wip_limits": {"interactive": 4},
    "stage_wip_limits": {"0": 8, "1": 7, "2": 7}
  }
}
```

Omitted limits are unbounded. A limit of zero pauses new matching dispatches.
Lowering a limit never preempts a lease that is already active. Malformed JSON
or an invalid schema leaves the most recent valid configuration in effect.

`VLLM_OMNI_RUNTIME_METRICS_DIR` enables an atomic, bounded-cardinality JSON
snapshot containing queue lengths, active leases, blocked reasons, dispatch
attempts, failures, cancellations, and queue-wait totals. Poll and snapshot
intervals can be set with `VLLM_OMNI_RUNTIME_CONTROL_INTERVAL_S` and
`VLLM_OMNI_RUNTIME_METRICS_INTERVAL_S`.

## Request metadata

The Python API accepts `request_class`, `request_path`, and
`first_output_deadline_s`. The OpenAI chat and speech endpoints expose the same
metadata through these optional headers:

- `x-vllm-omni-request-class`
- `x-vllm-omni-request-path`
- `x-vllm-omni-first-output-deadline-ms`

The deadline is converted once, at request arrival, to an absolute monotonic
deadline. EDF is stable FIFO for equal deadlines; requests without deadlines
sort after requests with deadlines. No request is rejected merely because its
deadline has passed.
