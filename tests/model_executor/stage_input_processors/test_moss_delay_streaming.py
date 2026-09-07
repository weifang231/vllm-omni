# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests of cumulative delay snapshots and the codec payload boundary."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
NQ = 32
PAD = 1024


@pytest.fixture
def processor(monkeypatch):
    # Exercise real torch and payload structs without loading a GPU engine.
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    for name, path in [
        ("vllm_omni.data_entry_keys", ROOT / "vllm_omni/data_entry_keys.py"),
        ("_moss_delay_processor_test", ROOT / "vllm_omni/model_executor/stage_input_processors/moss_tts.py"),
    ]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def _manager(**extra):
    return SimpleNamespace(connector=SimpleNamespace(config={"extra": extra}), request_payload={})


def _request(external="external", internal="internal", **overrides):
    entries = {key: SimpleNamespace(list_data=[value]) for key, value in overrides.items()}
    return SimpleNamespace(
        request_id=internal,
        external_req_id=external,
        additional_information=SimpleNamespace(entries=entries),
    )


def _codes(frames, offset=0):
    return (torch.arange(frames * NQ).reshape(frames, NQ) + offset) % PAD


def _delay(codes):
    delayed = torch.full((len(codes) + NQ - 1, NQ), PAD, dtype=torch.long)
    for q in range(NQ):
        delayed[q : q + len(codes), q] = codes[:, q]
    return delayed


def _call(processor, manager, request, snapshot=None, finished=False):
    output = None if snapshot is None else {"codes": {"audio": snapshot}}
    return processor.talker2codec_delay_streaming_async_chunk(manager, output, request, finished)


def _rows(payload):
    return torch.tensor(payload.codes.audio, dtype=torch.long).reshape(NQ, -1).transpose(0, 1)


def _join(payloads):
    return torch.cat([_rows(payload) for payload in payloads], dim=0)


def test_first_packet_waits_for_complete_codebooks_then_uses_eight_frames(processor):
    expected = _codes(19)
    leading = 5
    delayed = torch.cat([torch.full((leading, NQ), PAD), _delay(expected)])
    manager, request = _manager(), _request()
    packets = []
    boundaries = []
    for end in range(1, len(delayed) + 1):
        packet = _call(processor, manager, request, delayed[:end])
        if packet is not None:
            packets.append(packet)
            boundaries.append(end)
    packets.append(_call(processor, manager, request, finished=True))
    assert boundaries == [leading + NQ - 1 + 8, leading + NQ - 1 + 16]
    assert [packet.meta.codec_chunk_frames for packet in packets] == [8, 8, 3]
    assert torch.equal(_join(packets), expected)
    assert not manager.request_payload


@pytest.mark.parametrize("stride", [1, 3, 17, 1000])
def test_segments_and_padding_preserve_full_sequence_codes(processor, stride):
    first, second = _codes(9), _codes(21, offset=123)
    delayed = torch.cat(
        [
            torch.full((11, NQ), PAD),
            _delay(first),
            torch.full((5, NQ), PAD),
            _delay(second),
            torch.full((9, NQ), PAD),
        ]
    )
    manager, request = _manager(), _request()
    packets = []
    for end in range(stride, len(delayed), stride):
        packet = _call(processor, manager, request, delayed[:end])
        if packet is not None:
            packets.append(packet)
    packets.append(_call(processor, manager, request, delayed, finished=True))
    assert torch.equal(_join(packets), torch.cat([first, second]))
    assert all(packet.meta.req_id == [request.external_req_id] for packet in packets)
    assert not manager.request_payload


@pytest.mark.parametrize("rows, complete", [(0, 0), (31, 0), (32, 1), (33, 2)])
def test_exact_nq_boundary_and_incomplete_terminal_tail(processor, rows, complete):
    expected = _codes(2)
    manager, request = _manager(), _request()
    packet = _call(processor, manager, request, _delay(expected)[:rows], finished=True)
    assert torch.equal(_rows(packet), expected[:complete])
    assert packet.meta.finished.item() and packet.meta.stream_finished.item()
    assert not manager.request_payload


def test_all_pad_filter_preserves_partially_padded_frames(processor):
    expected = _codes(4)
    expected[1] = PAD
    expected[2, :16] = PAD
    packet = _call(processor, _manager(), _request(), _delay(expected), finished=True)
    assert torch.equal(_rows(packet), expected[[0, 2, 3]])


@pytest.mark.parametrize("terminal_data", [None, torch.empty((0, NQ), dtype=torch.long)])
def test_repeated_snapshot_and_empty_terminal_do_not_duplicate_audio(processor, terminal_data):
    expected = _codes(16)
    delayed = _delay(expected)
    manager, request = _manager(), _request()
    first = _call(processor, manager, request, delayed)
    second = _call(processor, manager, request, delayed)
    assert _call(processor, manager, request, delayed) is None
    terminal = _call(processor, manager, request, terminal_data, finished=True)
    assert torch.equal(_join([first, second, terminal]), expected)
    assert terminal.codes.audio == []
    assert terminal.request_id == "external"
    assert terminal.meta.req_id == ["external"]
    assert terminal.meta.codec_streaming
    assert terminal.meta.codec_chunk_frames == terminal.meta.code_flat_numel == 0
    assert terminal.meta.stream_finished.item()
    assert not manager.request_payload


def test_no_audio_still_has_explicit_terminal_payload(processor):
    manager = _manager()
    packet = _call(processor, manager, _request(), torch.full((80, NQ), PAD), finished=True)
    assert packet.codes.audio == []
    assert packet.meta.finished.item()
    assert not manager.request_payload


def test_only_new_snapshot_rows_are_copied_and_tail_storage_is_bounded(processor, monkeypatch):
    sizes = []
    original_to = torch.Tensor.to

    def traced_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            sizes.append(tuple(tensor.shape))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", traced_to)
    manager, request = _manager(), _request()
    delayed = _delay(_codes(100))
    for end in range(1, len(delayed) + 1):
        _call(processor, manager, request, delayed[:end])
        state = manager.request_payload["external"]
        assert len(state.delayed_tail) <= NQ - 1
        assert state.delayed_tail.untyped_storage().nbytes() == state.delayed_tail.numel() * 8
        assert len(state.pending) < 8
    assert sizes == [(1, NQ)] * len(delayed)
    _call(processor, manager, request, finished=True)


def test_request_chunk_overrides_are_frozen_for_request_lifetime(processor):
    manager = _manager(initial_codec_chunk_frames=2, codec_chunk_frames=5)
    request = _request(initial_codec_chunk_frames=3, codec_chunk_frames=4)
    expected = _codes(10)
    first = _call(processor, manager, request, _delay(expected))
    request.additional_information.entries["codec_chunk_frames"].list_data = [100]
    second = _call(processor, manager, request)
    terminal = _call(processor, manager, request, finished=True)
    assert [packet.meta.codec_chunk_frames for packet in [first, second, terminal]] == [3, 4, 3]
    assert torch.equal(_join([first, second, terminal]), expected)


@pytest.mark.parametrize("steady_frames", [16, 32])
def test_larger_later_chunks_preserve_first_audio_boundary_and_every_code(processor, steady_frames):
    manager = _manager(initial_codec_chunk_frames=8, codec_chunk_frames=steady_frames)
    request = _request()
    expected = _codes(77)
    delayed = _delay(expected)
    packets, boundaries = [], []
    for end in range(1, len(delayed) + 1):
        packet = _call(processor, manager, request, delayed[:end])
        if packet is not None:
            packets.append(packet)
            boundaries.append(end)
    packets.append(_call(processor, manager, request, finished=True))
    assert boundaries == list(range(NQ - 1 + 8, len(delayed) + 1, steady_frames))
    assert packets[0].meta.codec_chunk_frames == 8
    assert all(packet.meta.codec_chunk_frames == steady_frames for packet in packets[1:-1])
    assert packets[-1].meta.stream_finished.item()
    assert torch.equal(_join(packets), expected)
    assert not manager.request_payload


def test_external_identity_is_stable_across_internal_ids_and_cleanup(processor):
    manager = _manager()
    first_req = _request("a", "a-stage0")
    second_req = _request("b", "b-stage0")
    _call(processor, manager, first_req, _delay(_codes(8)))
    _call(processor, manager, second_req, _delay(_codes(8, offset=33)))
    assert set(manager.request_payload) == {"a", "b"}
    # cleanup_sender() clears request_payload by external ID, including aborts.
    manager.request_payload.pop("a")
    first_req = _request("a", "a-new-stage0")
    packet = _call(processor, manager, first_req, _delay(_codes(3, offset=99)), finished=True)
    assert torch.equal(_rows(packet), _codes(3, offset=99))
    assert packet.meta.req_id == ["a"]
    assert set(manager.request_payload) == {"b"}
    terminal = _call(processor, manager, _request("b", "b-another-id"), finished=True)
    assert terminal.meta.req_id == ["b"]
    assert not manager.request_payload
    assert not hasattr(manager, "_moss_tts_state")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "8", float("inf"), float("nan")])
def test_invalid_chunk_size_is_rejected(processor, value):
    with pytest.raises(ValueError, match="positive integer"):
        _call(processor, _manager(codec_chunk_frames=value), _request())


@pytest.mark.parametrize("values", [None, [], [1, 2]])
def test_invalid_request_override_is_rejected(processor, values):
    request = _request(codec_chunk_frames=8)
    request.additional_information.entries["codec_chunk_frames"].list_data = values
    with pytest.raises(ValueError, match="one frame count"):
        _call(processor, _manager(), request)


@pytest.mark.parametrize("shape", [(5, 31), (4, NQ)])
def test_snapshot_width_change_or_nonempty_rewind_is_rejected(processor, shape):
    manager, request = _manager(), _request()
    _call(processor, manager, request, torch.zeros((5, NQ), dtype=torch.long))
    with pytest.raises(ValueError, match="changed NQ|shrank"):
        _call(processor, manager, request, torch.zeros(shape, dtype=torch.long))


def test_flat_snapshot_is_rejected(processor):
    with pytest.raises(ValueError, match="must be \\[T, NQ\\]"):
        _call(processor, _manager(), _request(), torch.zeros(NQ, dtype=torch.long))


@pytest.mark.parametrize("enabled", [False, True])
def test_deploy_flag_selects_streaming_or_original_terminal_path(processor, enabled):
    manager, request = _manager(moss_v1_streaming=enabled), _request()
    expected = _codes(8)
    mm = {"codes": {"audio": _delay(expected)}}
    packet = processor.talker2codec_delay_async_chunk(manager, mm, request)
    assert (packet is not None) == enabled
    terminal = processor.talker2codec_delay_async_chunk(manager, mm, request, True)
    assert torch.equal(_join([packet, terminal] if enabled else [terminal]), expected)
    assert not manager.request_payload
    if not enabled:
        assert not manager._moss_tts_state


def test_nonboolean_deploy_flag_is_rejected(processor):
    with pytest.raises(ValueError, match="moss_v1_streaming must be a boolean"):
        processor.talker2codec_delay_async_chunk(_manager(moss_v1_streaming="false"), None, _request())
