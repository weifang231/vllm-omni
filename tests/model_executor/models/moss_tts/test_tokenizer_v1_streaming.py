# SPDX-License-Identifier: Apache-2.0
"""CPU checks for v1 incremental decoding; runnable without importing vLLM."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_tokenizer():
    source = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/moss_tts/audio_tokenizer.py"
    spec = importlib.util.spec_from_file_location("moss_v1_tokenizer_tested", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tokenizer = load_tokenizer()


def tiny_model(quantizer_type="rlfq"):
    def transformer(input_dim, output_dim):
        spec = tokenizer._transformer_block(input_dim, output_dim, 8, 2, 2)
        spec.update(dim_feedforward=16, layer_scale=None)
        return spec

    config = tokenizer.MossAudioTokenizerConfig(
        sampling_rate=8,
        downsample_rate=2,
        causal_transformer_context_duration=1,
        encoder_kwargs=[{"module_type": "PatchedPretransform", "patch_size": 2}, transformer(2, 4)],
        decoder_kwargs=[
            transformer(4, 8),
            {"module_type": "PatchedPretransform", "patch_size": 2},
            transformer(4, 2),
            {"module_type": "PatchedPretransform", "patch_size": 2},
        ],
        quantizer_kwargs={
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 2,
            "codebook_size": 8,
            "codebook_dim": 2,
            "quantizer_type": quantizer_type,
        },
    )
    torch.manual_seed(42)
    model = tokenizer.MossAudioTokenizerModel(config).eval()
    # Avoid near-silent default initialization hiding errors behind absolute tolerances.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            elif name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.normal_(std=0.2)
    return model


def step(model, chunks, slots, valid=None):
    max_length = max((chunk.shape[-1] for chunk in chunks), default=0)
    codes = torch.full((2, len(chunks), max_length), 999, dtype=torch.long)
    for row, chunk in enumerate(chunks):
        codes[:, row, : chunk.shape[-1]] = chunk
    return model.decode_streaming_batch(
        codes,
        torch.tensor([chunk.shape[-1] for chunk in chunks], dtype=torch.long),
        torch.tensor(slots, dtype=torch.long),
        torch.tensor(valid if valid is not None else [True] * len(slots), dtype=torch.bool),
    )


class TestV1StreamingTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uneven_chunks_match_full_decode_past_context(self):
        for quantizer_type in ("rlfq", "rvq"):
            for chunks in ([1] * 19, [3, 5, 1, 10], [19]):
                with self.subTest(quantizer_type=quantizer_type, chunks=chunks):
                    model = tiny_model(quantizer_type)
                    codes = torch.randint(0, 8, (2, 19))
                    expected = model.batch_decode([codes])
                    self.assertGreater(float(expected.audio.square().mean().sqrt()), 1e-3)
                    before = {key: value.clone() for key, value in model.state_dict().items()}
                    model.initialize_decoder_state_pool(1)
                    outputs = []
                    offset = 0
                    for length in chunks:
                        output = step(model, [codes[:, offset : offset + length]], [0])
                        self.assertEqual(output.audio_lengths.tolist(), [length * 4])
                        outputs.append(output.audio)
                        offset += length
                        for module, state in model._decoder_state_pool[0].items():
                            self.assertLessEqual(state.keys.shape[2], module.context - 1)
                            self.assertEqual(state.keys.untyped_storage().nbytes(), state.keys.numel() * 4)
                    actual = torch.cat(outputs, dim=-1)
                    torch.testing.assert_close(actual, expected.audio, atol=2e-5, rtol=2e-5)
                    nrmse = (actual - expected.audio).square().mean().sqrt() / expected.audio.square().mean().sqrt()
                    self.assertLess(float(nrmse), 1e-4)
                    # Pool initialization/use must not add checkpoint keys or alter offline output.
                    after = model.batch_decode([codes])
                    torch.testing.assert_close(after.audio, expected.audio, atol=0, rtol=0)
                    self.assertEqual(before.keys(), model.state_dict().keys())
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_interleaved_slots_ignore_padding_and_preserve_unselected_slot(self):
        model = tiny_model()
        a, b = torch.randint(0, 8, (2, 13)), torch.randint(0, 8, (2, 11))
        expected = [model.batch_decode([x]).audio[0] for x in (a, b)]
        model.initialize_decoder_state_pool(2, scratch_capacity=1)
        first = step(model, [a[:, :3], b[:, :2]], [0, 2])
        second = step(model, [b[:, 2:7], a[:, 3:7]], [2, 0])
        a_last = step(model, [a[:, 7:]], [0])
        b_last = step(model, [b[:, 7:]], [2])
        actual_a = torch.cat((first.audio[0, :, :12], second.audio[1, :, :16], a_last.audio[0]), dim=-1)
        actual_b = torch.cat((first.audio[1, :, :8], second.audio[0], b_last.audio[0]), dim=-1)
        for actual, ref in zip((actual_a, actual_b), expected):
            torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
        self.assertEqual(first.audio_lengths.tolist(), [12, 8])
        self.assertEqual(second.audio_lengths.tolist(), [20, 16])
        self.assertTrue(torch.equal(first.audio[1, :, 8:], torch.zeros(1, 4)))
        self.assertEqual(model._decoder_state_pool[1], {})

    def test_reset_reuses_one_slot_without_resetting_other_requests(self):
        model = tiny_model()
        a, b = torch.randint(0, 8, (2, 9)), torch.randint(0, 8, (2, 12))
        ref_a, ref_b = (model.batch_decode([x]).audio[0] for x in (a, b))
        model.initialize_decoder_state_pool(2)
        first = step(model, [a[:, :4], b[:, :5]], [0, 1])
        model.reset_decoder_state_slots(torch.tensor([0]))
        final = step(model, [b[:, 5:], a], [1, 0])
        torch.testing.assert_close(final.audio[1], ref_a, atol=2e-5, rtol=2e-5)
        actual_b = torch.cat((first.audio[1], final.audio[0, :, :28]), dim=-1)
        torch.testing.assert_close(actual_b, ref_b, atol=2e-5, rtol=2e-5)

    def test_empty_and_invalid_rows_do_not_advance_state(self):
        model = tiny_model()
        codes = torch.randint(0, 8, (2, 9))
        expected = model.batch_decode([codes]).audio[0]
        model.initialize_decoder_state_pool(2)
        first = step(model, [codes[:, :3]], [0])
        offsets = [state.position_offset for state in model._decoder_state_pool[0].values()]
        invalid = step(model, [torch.full((2, 4), 999)], [-1], [False])
        self.assertEqual(invalid.audio.shape, (1, 1, 0))
        self.assertEqual(invalid.audio_lengths.tolist(), [0])
        empty = step(model, [codes[:, :0], codes[:, :0]], [0, 1])
        self.assertEqual(empty.audio.shape, (2, 1, 0))
        self.assertEqual(empty.audio_lengths.tolist(), [0, 0])
        self.assertEqual(offsets, [state.position_offset for state in model._decoder_state_pool[0].values()])
        self.assertEqual(model._decoder_state_pool[1], {})
        final = step(model, [codes[:, 3:]], [0])
        torch.testing.assert_close(torch.cat((first.audio[0], final.audio[0]), dim=-1), expected, atol=2e-5, rtol=2e-5)

    def test_validation_and_pool_lifecycle(self):
        model = tiny_model()
        codes = torch.zeros((2, 2), dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            step(model, [codes], [0])
        with self.assertRaises(ValueError):
            model.initialize_decoder_state_pool(0)
        model.initialize_decoder_state_pool(2)
        with self.assertRaisesRegex(RuntimeError, "existing"):
            model.initialize_decoder_state_pool(2)
        for chunks, slots in (([codes, codes], [0, 0]), ([codes], [2])):
            with self.assertRaises(ValueError):
                step(model, chunks, slots)
            self.assertEqual(model._decoder_state_pool, [{}, {}])
        with self.assertRaises(ValueError):
            model.reset_decoder_state_slots(torch.tensor([-1]))
        model.close_decoder_state_pool()
        self.assertIsNone(model._decoder_state_pool)
        model.initialize_decoder_state_pool(1)
        self.assertEqual(model._decoder_state_pool, [{}])
        self.assertFalse(model.supports_streaming_cudagraph)
        self.assertTrue(model.requires_streaming_opt_in)

    def test_bfloat16_decoder_and_quantizer_dtypes(self):
        for quantizer_float32 in (True, False):
            with self.subTest(quantizer_float32=quantizer_float32):
                model = tiny_model().to(torch.bfloat16)
                if quantizer_float32:
                    model.quantizer.float()
                codes = torch.randint(0, 8, (2, 13))
                reference = model.batch_decode([codes]).audio
                model.initialize_decoder_state_pool(1)
                first = step(model, [codes[:, :3]], [0])
                last = step(model, [codes[:, 3:]], [0])
                actual = torch.cat((first.audio, last.audio), dim=-1)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, atol=0.025, rtol=0.025)


if __name__ == "__main__":
    unittest.main()
