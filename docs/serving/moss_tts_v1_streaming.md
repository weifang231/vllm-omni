# Experimental MOSS-TTS v1 Streaming

`OpenMOSS-Team/MOSS-TTS` can decode audio incrementally with its original
v1 audio tokenizer through an explicit deployment opt-in. The experimental
configuration is
[`moss_tts_v1_streaming.yaml`](../../vllm_omni/deploy/moss_tts_v1_streaming.yaml).
It uses eight-frame chunks and an FP32 codec decoder. Existing deployments
without `moss_v1_streaming: true` retain terminal, full-sequence decoding.

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
| `connectors.shm.extra.codec_chunk_frames` | `8` | Number of complete frames in subsequent nonterminal chunks. |
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
eight-frame chunks and flushes any remaining complete frames at completion.

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
