# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio

import pytest

from vllm_omni.entrypoints.openai.playback_start import (
    MAX_PLAYBACK_BUFFER_MS,
    PLAYBACK_DEADLINE_EVENT,
    PlaybackStartBuffer,
    PlaybackStartConfig,
    iterate_with_playback_deadline,
    playback_start_config_from_headers,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_playback_headers_require_explicit_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = {
        "X-VLLM-OMNI-PLAYBACK-BUFFER-MS": "300",
        "x-vllm-omni-first-output-deadline-ms": "750",
    }
    assert playback_start_config_from_headers(headers, request_start_s=10.0) is None

    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    assert playback_start_config_from_headers(headers, request_start_s=10.0) == PlaybackStartConfig(
        target_ms=300.0,
        deadline_monotonic_s=10.75,
    )


def test_deadline_without_target_does_not_enable_playback_buffer() -> None:
    config = playback_start_config_from_headers(
        {"x-vllm-omni-first-output-deadline-ms": "750"},
        request_start_s=10.0,
        trusted=True,
    )
    assert config is None


@pytest.mark.parametrize("value", ["nan", "inf", "-1", str(MAX_PLAYBACK_BUFFER_MS + 1)])
def test_playback_target_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError, match="playback-buffer-ms"):
        playback_start_config_from_headers(
            {"x-vllm-omni-playback-buffer-ms": value},
            request_start_s=0.0,
            trusted=True,
        )


def test_buffer_counts_pcm_frames_across_sample_rates_exactly() -> None:
    now = [10.0]
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(target_ms=150.0),
        clock=lambda: now[0],
    )
    # 100 mono PCM16 frames at 1 kHz = 100 ms.
    assert (
        buffer.add_pcm(
            "a",
            pcm_bytes=b"\x00" * 200,
            sample_rate=1000,
            num_channels=1,
            prefix_items=("header",),
        )
        == ()
    )
    now[0] = 10.03
    # 100 stereo PCM16 frames at 2 kHz = 50 ms.
    released = buffer.add_pcm(
        "b",
        pcm_bytes=b"\x00" * 400,
        sample_rate=2000,
        num_channels=2,
    )
    assert released == ("header", "a", "b")
    assert buffer.telemetry(status="ok") == {
        "status": "ok",
        "target_ms": 150.0,
        "buffered_audio_ms": 150.0,
        "hold_ms": 30.0,
        "release_reason": "target",
        "deadline_fallback": False,
    }


def test_eos_flushes_short_audio_and_error_discards_it() -> None:
    buffer = PlaybackStartBuffer(PlaybackStartConfig(target_ms=500.0), clock=lambda: 1.0)
    assert buffer.add_pcm("pcm", pcm_bytes=b"\x00" * 200, sample_rate=1000, num_channels=1) == ()
    assert buffer.finish() == ("pcm",)
    assert buffer.telemetry(status="ok")["release_reason"] == "eos"

    failed = PlaybackStartBuffer(PlaybackStartConfig(target_ms=500.0), clock=lambda: 1.0)
    assert failed.add_pcm("pcm", pcm_bytes=b"\x00" * 200, sample_rate=1000, num_channels=1) == ()
    failed.terminate("error")
    assert failed.finish() == ()
    assert failed.telemetry(status="error")["release_reason"] == "error"


@pytest.mark.asyncio
async def test_deadline_event_does_not_cancel_pending_engine_pull() -> None:
    allow_result = asyncio.Event()
    generator_closed = asyncio.Event()

    async def engine_results():
        try:
            await allow_result.wait()
            yield "result"
        finally:
            generator_closed.set()

    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(
            target_ms=500.0,
            deadline_monotonic_s=asyncio.get_running_loop().time() + 0.01,
        ),
        clock=asyncio.get_running_loop().time,
    )
    stream = iterate_with_playback_deadline(engine_results(), buffer)
    assert await asyncio.wait_for(anext(stream), timeout=1.0) is PLAYBACK_DEADLINE_EVENT
    buffer.release_deadline()
    allow_result.set()
    assert await asyncio.wait_for(anext(stream), timeout=1.0) == "result"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert generator_closed.is_set()


@pytest.mark.asyncio
async def test_deadline_iterator_cancels_pending_pull_when_client_cancels() -> None:
    engine_waiting = asyncio.Event()
    generator_closed = asyncio.Event()

    async def engine_results():
        try:
            engine_waiting.set()
            await asyncio.Event().wait()
            yield "unreachable"  # pragma: no cover
        finally:
            generator_closed.set()

    loop = asyncio.get_running_loop()
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(
            target_ms=500.0,
            deadline_monotonic_s=loop.time() + 60.0,
        ),
        clock=loop.time,
    )
    stream = iterate_with_playback_deadline(engine_results(), buffer)
    pending_delivery = asyncio.create_task(anext(stream))
    await asyncio.wait_for(engine_waiting.wait(), timeout=1.0)
    pending_delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_delivery
    assert generator_closed.is_set()
