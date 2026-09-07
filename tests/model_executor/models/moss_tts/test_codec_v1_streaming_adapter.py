# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual adapter classes without importing GPU-only vLLM modules."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn


def load_adapter():
    source = Path(__file__).resolve().parents[4] / (
        "vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py"
    )
    tree = ast.parse(source.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"_MossCodecStreamSession", "MossTTSCodecDecoder"}
    )
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {
        "torch": torch,
        "nn": nn,
        "logger": logging.getLogger(__name__),
        "OmniOutput": SimpleNamespace,
        "CUDAGraphStreamingDecoderWrapper": Mock(side_effect=AssertionError("Unexpected streaming CUDA graph")),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


adapter = load_adapter()
Session = adapter["_MossCodecStreamSession"]
Decoder = adapter["MossTTSCodecDecoder"]


class FakeCodec(nn.Module):
    requires_streaming_opt_in = True
    supports_streaming_cudagraph = False
    downsample_rate = 2  # Deliberately differs from the decoded three samples per frame.

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(codebook_size=8)
        self.states = None
        self.reset_calls = []
        self.offline_calls = []
        self.stream_calls = []
        self.initialized = None

    def initialize_decoder_state_pool(self, capacity, scratch_capacity=0):
        self.initialized = (capacity, scratch_capacity)
        self.states = [0] * (capacity + scratch_capacity)

    def reset_decoder_state_slots(self, slots):
        self.reset_calls.append(tuple(slots.tolist()))
        for slot in slots.tolist():
            self.states[slot] = 0

    def close_decoder_state_pool(self):
        self.states = None

    def decode_streaming_batch(self, codes, lengths, slots, valid):
        self.stream_calls.append((codes.clone(), slots.tolist()))
        audio = torch.full((len(slots), 1, codes.shape[-1] * 3 + 7), -999.0)
        output_lengths = []
        for row, slot in enumerate(slots.tolist()):
            length = int(lengths[row]) if bool(valid[row]) else 0
            waveform = codes[0, row, :length].repeat_interleave(3).float()
            audio[row, 0, : len(waveform)] = waveform
            self.states[slot] += length
            output_lengths.append(len(waveform))
        return SimpleNamespace(audio=audio, audio_lengths=torch.tensor(output_lengths))

    def batch_decode(self, codes_list, num_quantizers):
        self.offline_calls.append([codes.clone() for codes in codes_list])
        waveform = codes_list[0][0].repeat_interleave(3).float()
        audio = torch.cat((waveform, torch.full((7,), -999.0))).reshape(1, 1, -1)
        return SimpleNamespace(audio=audio, audio_lengths=torch.tensor([waveform.numel()]))


def make_decoder(*, opt_in=False, eager=True):
    decoder = Decoder.__new__(Decoder)
    nn.Module.__init__(decoder)
    decoder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=eager,
            stage_connector_config={"extra": {"moss_v1_streaming": opt_in}},
        )
    )
    decoder._codec = FakeCodec()
    decoder._n_vq = 2
    decoder._n_channels = 1
    decoder._sr_tensor = torch.tensor(24000, dtype=torch.int32)
    decoder._async_chunk = True
    decoder._cuda_graph_wrapper = None
    decoder._stream_session = None
    decoder._stream_state_capacity = 3
    decoder._stream_max_step_frames = 4
    decoder._stream_req_slots = {}
    decoder._stream_fallback_buffers = {}
    decoder._streaming_graph_batch_sizes = [1, 2]
    decoder._streaming_graph_frame_sizes = [4]
    return decoder


def codes(values):
    return torch.tensor([values, values], dtype=torch.long)


def info(request_id, finished):
    return {"meta": {"req_id": [request_id], "stream_finished": [finished]}}


class TestV1CodecAdapter(unittest.TestCase):
    def test_default_opt_out_buffers_until_terminal_and_trims_actual_length(self):
        decoder = make_decoder()
        self.assertFalse(decoder._codec_supports_streaming())
        self.assertEqual(decoder._decode_streaming_batch([(0, "a", codes([1, 2]), False)]), {})
        self.assertIsNone(decoder._stream_session)
        self.assertEqual(decoder._codec.offline_calls, [])
        output = decoder._decode_streaming_batch([(0, "a", codes([3]), True)])
        torch.testing.assert_close(output[0], torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3.0]]))
        self.assertEqual(len(decoder._codec.offline_calls), 1)
        self.assertEqual(decoder._stream_fallback_buffers, {})
        self.assertIsNone(decoder._codec.states)

    def test_missing_opt_in_key_stays_terminal_and_v2_needs_no_opt_in(self):
        decoder = make_decoder()
        decoder.vllm_config.model_config.stage_connector_config = {"extra": {}}
        self.assertFalse(decoder._codec_supports_streaming())
        decoder._codec.requires_streaming_opt_in = False
        self.assertTrue(decoder._codec_supports_streaming())

    def test_decoder_precision_defaults_and_explicit_connector_settings(self):
        decoder = make_decoder()
        self.assertEqual(decoder._decoder_dtype(torch.device("cuda")), torch.bfloat16)
        self.assertEqual(decoder._decoder_dtype(torch.device("cpu")), torch.float32)
        for name, expected in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
            for connector in (
                {"extra": {"moss_v1_decoder_dtype": name}},
                {"moss_v1_decoder_dtype": name},
                SimpleNamespace(extra={"moss_v1_decoder_dtype": name}),
            ):
                with self.subTest(precision=name, connector=connector):
                    decoder.vllm_config.model_config.stage_connector_config = connector
                    self.assertEqual(decoder._decoder_dtype(torch.device("cuda:0")), expected)
                    self.assertEqual(decoder._decoder_dtype(torch.device("cpu")), torch.float32)
        decoder.vllm_config.model_config.stage_connector_config = None
        self.assertEqual(decoder._decoder_dtype(torch.device("cuda")), torch.bfloat16)

    def test_invalid_precision_fails_explicitly_on_every_device(self):
        decoder = make_decoder()
        for value in ("float16", "fp32", "", None, False, 32, [], {"dtype": "float32"}):
            decoder.vllm_config.model_config.stage_connector_config = {"extra": {"moss_v1_decoder_dtype": value}}
            for device in ("cpu", "cuda"):
                with self.subTest(value=value, device=device):
                    with self.assertRaisesRegex(ValueError, "Unsupported moss_v1_decoder_dtype"):
                        decoder._decoder_dtype(torch.device(device))

    def test_eager_only_codec_never_initializes_streaming_graph(self):
        decoder = make_decoder(opt_in=True, eager=False)
        decoder._enable_non_streaming_decoder_cudagraph = Mock(side_effect=AssertionError("Wrong graph path"))
        decoder._configure_decoder_cudagraph(torch.device("cpu"))
        self.assertEqual(decoder._streaming_graph_batch_sizes, [])
        decoder._enable_non_streaming_decoder_cudagraph.assert_not_called()
        codec = FakeCodec()
        real_arange = torch.arange

        def cpu_arange(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_arange(*args, **kwargs)

        # Simulate device selection only, so removing the capability gate would call the failing graph factory.
        with patch.object(codec, "parameters", return_value=iter([SimpleNamespace(device=torch.device("cuda"))])):
            with patch.object(torch, "arange", side_effect=cpu_arange):
                session = Session(
                    codec, state_capacity=3, n_vq=2, vllm_config=None, graph_batch_sizes=[1, 2], graph_frame_sizes=[4]
                )
        self.assertEqual(codec.initialized, (3, 0))
        self.assertIsNone(session._cudagraph_wrapper)
        adapter["CUDAGraphStreamingDecoderWrapper"].assert_not_called()

    def test_session_uses_audio_lengths_and_resets_only_terminal_slot(self):
        codec = FakeCodec()
        session = Session(codec, state_capacity=2, n_vq=2, vllm_config=None)
        a, b = session.acquire(), session.acquire()
        result = session.step({b: codes([3, 4]), a: codes([1, 2])}, terminal_slots={b})
        self.assertEqual(result[a].shape, (1, 6))
        torch.testing.assert_close(result[a], torch.tensor([[1, 1, 1, 2, 2, 2.0]]))
        self.assertEqual(codec.states, [2, 0])
        self.assertEqual(codec.reset_calls, [(b,)])
        session.release(b, state_already_reset=True)
        self.assertEqual(session._leased_slots, {a})
        self.assertEqual(session.acquire(), b)
        session.close()
        self.assertIsNone(codec.states)

    def test_metadata_false_list_does_not_finish_live_request(self):
        decoder = make_decoder(opt_in=True)
        output = decoder.forward(
            input_ids=codes([1, 2]).flatten(), runtime_additional_information=[info("a", False)], seq_token_counts=[4]
        )
        self.assertEqual(output.multimodal_outputs["model_outputs"][0].shape, (6,))
        self.assertIn("a", decoder._stream_req_slots)
        slot = decoder._stream_req_slots["a"]
        self.assertEqual(decoder._codec.states[slot], 2)
        decoder.forward(
            input_ids=codes([3]).flatten(), runtime_additional_information=[info("a", True)], seq_token_counts=[2]
        )
        self.assertEqual(decoder._stream_req_slots, {})
        self.assertEqual(decoder._codec.states[slot], 0)

    def test_terminal_flags_are_explicit_booleans(self):
        for value in (None, False, 0, [False], (False,), torch.tensor(False), torch.tensor([False])):
            self.assertIs(Decoder._metadata_flag(value), False)
        for value in (True, 1, [True], torch.tensor(True)):
            self.assertIs(Decoder._metadata_flag(value), True)
        for value in ("false", "true", 2, [], [False, True], torch.tensor([False, True])):
            with self.assertRaises(ValueError):
                Decoder._metadata_flag(value)

    def test_scheduler_ids_override_transport_ids_for_abort_and_empty_terminal(self):
        decoder = make_decoder(opt_in=True)
        self.assertTrue(decoder.requires_request_ids)
        decoder.forward(
            input_ids=torch.cat((codes([1]).flatten(), codes([2]).flatten())),
            seq_token_counts=[2, 2],
            request_ids=["engine-a", "engine-b"],
            runtime_additional_information=[info("transport-a", False), info("transport-b", False)],
        )
        self.assertEqual(set(decoder._stream_req_slots), {"engine-a", "engine-b"})
        a = decoder._stream_req_slots["engine-a"]
        decoder.forward(
            input_ids=None,
            request_ids=["engine-a", "engine-b"],
            runtime_additional_information=[info("transport-a", False), info("transport-b", True)],
        )
        self.assertEqual(decoder._stream_req_slots, {"engine-a": a})
        decoder.on_requests_finished(["engine-a"])
        self.assertEqual(decoder._stream_req_slots, {})
        self.assertEqual(decoder._stream_session._leased_slots, set())

    def test_zero_input_terminal_releases_named_request_only(self):
        decoder = make_decoder(opt_in=True)
        decoder._decode_streaming_batch([(0, "a", codes([1]), False), (1, "b", codes([2]), False)])
        a, b = decoder._stream_req_slots["a"], decoder._stream_req_slots["b"]
        decoder.forward(input_ids=None, runtime_additional_information=[info("a", False), info("b", True)])
        self.assertEqual(decoder._stream_req_slots, {"a": a})
        self.assertEqual(decoder._codec.states[a], 1)
        self.assertEqual(decoder._codec.states[b], 0)
        self.assertEqual(decoder._stream_session._leased_slots, {a})
        decoder._decode_streaming_batch([(0, "a", codes([]), True)])
        self.assertEqual(decoder._stream_req_slots, {})
        self.assertEqual(decoder._stream_session._leased_slots, set())

    def test_mixed_zero_token_terminal_and_live_row_remain_separate(self):
        decoder = make_decoder(opt_in=True)
        decoder._decode_streaming_batch([(0, "a", codes([1]), False), (1, "b", codes([2]), False)])
        a, b = decoder._stream_req_slots["a"], decoder._stream_req_slots["b"]
        decoder.forward(
            input_ids=codes([3]).flatten(),
            seq_token_counts=[0, 2],
            runtime_additional_information=[info("a", True), info("b", False)],
        )
        self.assertEqual(decoder._stream_req_slots, {"b": b})
        self.assertEqual(decoder._codec.states[a], 0)
        self.assertEqual(decoder._codec.states[b], 2)

    def test_abort_and_duplicate_finish_do_not_release_other_requests(self):
        decoder = make_decoder(opt_in=True)
        decoder._decode_streaming_batch([(0, "a", codes([1]), False), (1, "b", codes([2]), False)])
        a, b = decoder._stream_req_slots["a"], decoder._stream_req_slots["b"]
        decoder._stream_fallback_buffers = {"a": [codes([1])], "b": [codes([2])]}
        decoder.on_requests_finished(["b", "unknown", "b"])
        self.assertEqual(decoder._stream_req_slots, {"a": a})
        self.assertEqual(set(decoder._stream_fallback_buffers), {"a"})
        self.assertEqual(decoder._stream_session._leased_slots, {a})
        self.assertEqual(decoder._codec.states[b], 0)
        self.assertEqual(decoder._codec.states[a], 1)

    def test_long_chunk_split_preserves_audio_and_terminal_cleanup(self):
        decoder = make_decoder(opt_in=True)
        values = [1, 2, 3, 4, 5, 6, 7]
        result = decoder._decode_streaming_batch([(0, "a", codes(values), True)])
        torch.testing.assert_close(result[0], torch.tensor(values).repeat_interleave(3).float().unsqueeze(0))
        self.assertEqual([entry[0].shape[-1] for entry in decoder._codec.stream_calls], [4, 3])
        self.assertEqual(decoder._stream_req_slots, {})
        self.assertEqual(decoder._stream_session._leased_slots, set())
        self.assertTrue(all(state == 0 for state in decoder._codec.states))


if __name__ == "__main__":
    unittest.main()
