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

The control layer is responsible for:

1. enforcing configured pipeline, path, class, and per-stage work-in-progress
   limits without preempting requests that already hold a lease;
2. ordering queued stage submissions by FIFO or absolute first-output deadline;
3. carrying observable request class, path, and deadline metadata from the
   public engine API through the orchestrator and downstream stage requests;
4. exposing bounded-cardinality queue, active-lease, dispatch, and wait-time
   telemetry for evaluation and controller feedback; and
5. optionally applying the paper's calibrated Erlang--empirical ingress score
   before a request first enters stage 0, with rechecks after queue/configuration
   changes and immediately before dispatch; and
6. for Qwen3-TTS HTTP streams, optionally holding client-visible PCM until a
   controller-selected startup buffer is available or the first-output
   deadline expires.

It does not implement the paper's dynamic-program optimizer or compute the
Brownian startup-buffer formula. The admission implementation is a model-based
score using operator-supplied calibration data; it is not a formal
out-of-sample guarantee. The playback adapter is the runtime mechanism to apply
a target computed by that controller.

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
    "stage_wip_limits": {"0": 8, "1": 7, "2": 7},
    "admission": {
      "enabled": true,
      "score_method": "erlang_empirical",
      "classes": {
        "interactive": {
          "effective_k": 4,
          "mu": 0.31,
          "service_samples_s": [0.72, 0.84, 1.05, 1.31],
          "gamma": 0.9
        }
      }
    }
  }
}
```

Omitted limits are unbounded. A limit of zero pauses new matching dispatches.
Lowering a limit never preempts a lease that is already active. Malformed JSON
or an invalid schema leaves the most recent valid configuration in effect.

Admission is disabled unless `queue_control.admission.enabled` is true. For
each configured class, `effective_k` is the class concurrency limit, `mu` is
the measured per-occupied-slot service rate at that limit, and
`service_samples_s` contains calibration samples from execution start to a
valid first output. `gamma` is the frozen threshold selected on disjoint
calibration data. A configured request without a deadline bypasses the score;
an unconfigured class is unchanged. A rejection is request-scoped and returns
HTTP 429 with error type `AdmissionRejectedError` before an HTTP response is
committed, while the server continues serving other requests. An SSE response
that has already started emits `speech.audio.error` with code 429 and the same
type. A raw-audio stream cannot change its HTTP status after headers have been
sent and therefore terminates with the exception instead. Admission requires
`policy: "edf"`; operators should also use mutually consistent class/global
limits so runtime ordering and available capacity match the fitted model.

`VLLM_OMNI_RUNTIME_METRICS_DIR` enables an atomic, bounded-cardinality JSON
snapshot containing queue lengths, active leases, blocked reasons, dispatch
attempts, failures, cancellations, queue-wait totals, admission counters,
reason counts, and the latest 128 admission decisions. Poll and snapshot
intervals can be set with `VLLM_OMNI_RUNTIME_CONTROL_INTERVAL_S` and
`VLLM_OMNI_RUNTIME_METRICS_INTERVAL_S`.

## Request metadata

The Python API accepts `request_class`, `request_path`, and
`first_output_deadline_s`. The OpenAI chat and speech endpoints expose the same
metadata through these optional headers:

- `x-vllm-omni-request-class`
- `x-vllm-omni-request-path`
- `x-vllm-omni-first-output-deadline-ms`

HTTP scheduling headers are ignored by default. Set
`VLLM_OMNI_TRUST_SCHEDULING_HEADERS=1` only when a trusted ingress proxy strips
caller-provided values and supplies authenticated class, path, and deadline
metadata. Enabling the gate directly on a public endpoint would let a client
claim another class or an earlier deadline.

The deadline is converted once, at request arrival, to an absolute monotonic
deadline. EDF is stable FIFO for equal deadlines; requests without deadlines
sort after requests with deadlines. With admission disabled, no request is
rejected merely because its deadline has passed. With calibrated admission
enabled for its class, an expired request is rejected with HTTP 429.

## Qwen3-TTS playback-start adapter

The HTTP raw-audio and SSE streaming paths recognize one additional trusted
header:

- `x-vllm-omni-playback-buffer-ms`

The header is ignored unless `VLLM_OMNI_TRUST_SCHEDULING_HEADERS=1`. When it is
present, Qwen3-TTS continues draining decoded audio from the engine but
withholds the WAV header and PCM chunks from the client until their exact PCM16
frame duration reaches the requested target. It then flushes the original
chunks in order and streams subsequent chunks normally. A clean end of stream
flushes a short utterance even when it never reaches the target.

If `x-vllm-omni-first-output-deadline-ms` is also present, expiration opens the
gate with whatever audio is available; if no chunk is available yet, the first
later chunk is delivered immediately. The pending engine pull is not cancelled
or restarted when this fallback fires. Targets must be finite, non-negative,
and at most 60 seconds. Each enabled request emits one bounded telemetry record
with the target, buffered audio duration at release, actual wall-clock hold,
release reason, and whether deadline fallback was used.

This interface deliberately accepts the selected target rather than deriving
one. The paper's Brownian rule can run in an external or future in-process
controller using calibrated generation drift and variance. The current adapter
does not apply to non-Qwen3-TTS models, non-streaming responses, or the
sentence-oriented WebSocket endpoint.
