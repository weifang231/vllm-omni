# Experimental MOSS-TTS v1 Streaming

`OpenMOSS-Team/MOSS-TTS` can decode audio incrementally with its original
v1 audio tokenizer through an explicit deployment opt-in. The experimental
configuration is
[`moss_tts_v1_streaming.yaml`](../../vllm_omni/deploy/moss_tts_v1_streaming.yaml).
It uses an eight-frame first chunk, 32-frame subsequent chunks, and an FP32
codec decoder. Existing deployments without `moss_v1_streaming: true` retain
terminal, full-sequence decoding.

This configuration targets the 8B delay-pattern MOSS-TTS v1 model and its
v1 codec. Other MOSS variants have different decoding paths.

## Start the server

Run from the repository root in the vLLM-Omni serving environment:

```bash
vllm serve OpenMOSS-Team/MOSS-TTS \
    --omni \
    --host 127.0.0.1 \
    --port 8091 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/moss_tts_v1_streaming.yaml
```

Both stages use GPU 0 within the process's visible devices. The supplied
configuration allows four stage-0 requests and four stage-1 requests.
For local checkpoints, pass the TTS checkpoint as the model argument and
set the codec path explicitly:

```bash
vllm serve /path/to/MOSS-TTS \
    --served-model-name OpenMOSS-Team/MOSS-TTS \
    --omni \
    --host 127.0.0.1 \
    --port 8091 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/moss_tts_v1_streaming.yaml \
    --hf-overrides '{"codec_model_name_or_path":"/path/to/MOSS-Audio-Tokenizer"}'
```

Use the [Speech API](speech_api.md) with `stream: true`,
`stream_format: "audio"`, and `response_format: "pcm"` to receive raw
audio as it becomes available. Voice-cloning requests provide `ref_audio`
and its `ref_text`. Setting HTTP streaming alone does not enable the
incremental codec; the deployment opt-in is also required.

## Configuration and capacity

The supplied YAML sets these fields:

| Setting | Value | Purpose |
|---|---|---|
| `async_chunk` | `true` | Transfer code chunks between stages while generation continues. |
| `connectors.shm.extra.moss_v1_streaming` | `true` | Enable incremental delay-pattern processing and the stateful v1 decoder together. |
| `connectors.shm.extra.moss_v1_decoder_dtype` | `float32` | Select the decoder precision that passed the current equivalence checks. |
| `connectors.shm.extra.initial_codec_chunk_frames` | `8` | Number of complete frames required for the first chunk. |
| `connectors.shm.extra.codec_chunk_frames` | `32` | Coalesce later audio to reduce repeated eager decoder calls. |
| `connectors.shm.extra.codec_left_context_frames` | `0` | Send new frames only; the decoder retains its own attention context. |
| Stage 1 `max_num_seqs` | `4` | Provide decoder state slots for live streams. |
| Stage 1 `enforce_eager` | `true` | Use the supported eager decoder path. |

A live stream retains its decoder slot between chunks, including while it
waits for more stage-0 output. Size stage-1 `max_num_seqs` for the maximum
number of simultaneously live codec streams, rather than only the number
of chunks executing at one instant. Keep it at least as large as the
stage-0 live-stream capacity; account for downstream streams that have not
finished when increasing concurrency. The supplied recipe uses four for
both stages. State is isolated by request and released on completion or
abort. Exhausting the pool is an explicit error.

The decoder retains causal KV state within each layer's configured context
window and keeps absolute rotary positions across chunks. It does not
redecode the complete prefix. This v1 implementation is eager-only and
advertises `supports_streaming_cudagraph = False`; CUDA graph capture sizes
inherited from other configurations do not enable streaming graph support.

## When the first audio becomes available

MOSS-TTS v1 uses 32 delayed codebooks. A complete codec frame requires the
corresponding row from every codebook, introducing 31 rows of delay. With
an initial chunk of eight frames, the earliest chunk can be assembled
after 31 delay rows plus eight complete, valid frames: 39 delayed rows in
the simple case without discarded padding. The processor then emits
32-frame chunks and flushes any remaining complete frames at completion.

These are audio-code generation rows. An eight-frame chunk does not imply
audio after eight text tokens or a fixed TTFA. Prefill, queueing, delayed
code generation, codec execution, and transport still contribute to TTFA.
Use the decoder's reported sample count and sample rate to determine audio
duration.

## Precision and validation status

Use explicit `moss_v1_decoder_dtype: float32` for this experimental path.
The original v1 checkpoint passed full-sequence versus chunked numerical
equivalence checks in FP32, including uneven chunks, interleaved requests,
context boundaries, invalid padded rows, and slot reuse. The codec worker
disables CUDA matmul and cuDNN TF32 when FP32 is selected. The stage-0 TTS
model retains its separately configured precision.

BF16 chunk equivalence failed the registered error limits on real
checkpoint inputs. Its numerical discrepancy remains unresolved; switching
the decoder to BF16 is not a validated optimization. Omitting the precision
field preserves the legacy CUDA default of BF16, so the streaming recipe
sets FP32 explicitly. CPU codec execution uses FP32.

Check the startup log for `MOSS codec decoder precision: torch.float32` and
confirm that serving produces incremental audio before treating a deployment
as a successful streaming smoke test. Numerical equivalence alone does not
establish perceptual quality or serving performance. Goodput and latency
measurements for this configuration must be reported separately.

For a matched terminal comparison, use the same checkpoints, precision,
stage capacities, and client workload, and set
`connectors.shm.extra.moss_v1_streaming: false`. The default terminal path
performs one full-sequence codec decode when generation finishes.

## Measured goodput and limits

On one GB300, three paired 60-second Poisson traces at 4 requests/s increased
Ours goodput from **1.622 to 2.156 requests/s (+32.9%)**. Every pair improved.
Both arms used the same FP32 codec, BF16 talker, stage capacities 4/4, and
fixed K=3 / gamma=0.50 admission policy. The workload repeated five English
SeedTTS voice-cloning texts, with a warm reference cache and at most 256
generated tokens per request. A good request met both scheduled-arrival
TTFA <=1 second and cumulative playback stall <=0.1 second.

| Metric, pooled over three arrival seeds | Terminal Ours | First 8 / subsequent 32 Ours |
|---|---:|---:|
| Good / offered / completed requests | 292 / 714 / 417 | 388 / 714 / 388 |
| Goodput (requests/s) | 1.622 | 2.156 |
| Good fraction of all offers | 40.90% | 54.34% |
| Completed-request TTFA p50 / p95 (s) | 0.916 / 1.323 | 0.642 / 0.758 |
| Completed-request E2E p50 (s) | 0.917 | 1.112 |
| Policy rejections / nonpolicy errors | 297 / 0 | 326 / 0 |

All completed requests in both arms had zero cumulative playback stall.
The paired-seed bootstrap interval for the goodput increase was
0.333–0.667 requests/s. Three arrival seeds over five repeated texts limit
generalization. Total completions decreased, so these results establish
an improvement in first-audio SLO goodput, not total generation throughput
or 90/90 service qualification.

A later Native follow-up reused the exact three arrival traces and ran
six fresh servers on the same physical GPU and host, with matching runtime,
weights, YAMLs, and packages. Native sent no scheduling headers and enabled
no admission control. Pooled Native goodput did **not** improve:

| Metric, pooled over the same three arrival traces | Terminal Native | First 8 / subsequent 32 Native |
|---|---:|---:|
| Good / offered / completed requests | 22 / 714 / 714 | 16 / 714 / 714 |
| Goodput (requests/s) | 0.122 | 0.089 |
| Good fraction of all offers | 3.08% | 2.24% |
| Scheduled TTFA p50 / p95 (s) | 3.133 / 10.632 | 9.127 / 23.543 |
| Scheduled E2E p50 (s) | 3.134 | 9.809 |
| Total drain beyond three 60-second offered windows (s) | 20.216 | 64.020 |
| Policy rejections / nonpolicy errors | 0 / 0 | 0 / 0 |

The observed pooled change was -27.3%. Per-seed good counts changed from
2 to 7, 16 to 4, and 4 to 5; the paired-seed bootstrap difference interval
was -0.200 to +0.083 requests/s. These mixed, small counts do not establish
a consistent goodput improvement or regression. Latency and drain were
worse in all three pairs. Every request completed and passed the cumulative
stall limit. All streaming-good requests arrived within the first 2.209
seconds of their traces, while terminal decoding also met the SLO later.
Client queue time averaged below 1 ms, so it does not explain the
multi-second delays.

Repeated eager codec work can increase server queueing under Native load;
the earlier codec microprofile and these latency measurements are consistent
with that mechanism. A consistent goodput gain was observed at the
admission-controlled Ours operating point. Native was measured later with
the Ours-selected cadence. The two paired campaigns are not a
contemporaneously interleaved four-arm
trial or a search for Native's best setting. The follow-up therefore does
not support using this recipe as a general Native throughput optimization.

Uniform eight-frame chunks were substantially slower under load: the
preceding calibration achieved 0.350 requests/s against 1.700 for matched
terminal Ours. Coalescing later frames reduces repeated eager decoder calls;
the implementation still decodes batch rows individually. BF16 equivalence
remains unresolved, and this FP32 comparison does not replace historical
BF16 results. Reproducing the reported goodput requires the frozen admission
control and client deadline headers in addition to this deployment YAML.

The serving-study archive is
`experiments/topconf_expansion_20260907/moss_v1_streaming_goodput_v1`.
Its `RESULTS.md`, frozen manifests, and raw records preserve the negative
candidate, correctness gates, selection, Native follow-up, and confirmation seeds
9707341, 9707342, and 9707343. Measured runtime Python sources match commit
`66c7dd8bff7b9e3eac7d154179024c5c6f0860a7`.
