# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio

import pytest

from vllm_omni.engine.queue_control import (
    RequestSchedulingMetadata,
    scheduling_kwargs_from_headers,
)
from vllm_omni.entrypoints.openai.playback_start import (
    MAX_PLAYBACK_BUFFER_MS,
    PLAYBACK_DEADLINE_GUARD_MS_HEADER,
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
        deadline_guard_ms=0.0,
    )


def test_playback_deadline_guard_uses_live_runtime_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    config = playback_start_config_from_headers(
        {
            "x-vllm-omni-playback-buffer-ms": "321",
            "x-vllm-omni-first-output-deadline-ms": "1000",
            PLAYBACK_DEADLINE_GUARD_MS_HEADER: "20",
        },
        request_start_s=10.0,
    )
    assert config is not None
    assert config.deadline_monotonic_s == 11.0
    assert config.deadline_guard_ms == 20.0

    now = [10.7]
    buffer = PlaybackStartBuffer(config, clock=lambda: now[0])
    assert buffer.seconds_until_deadline() == pytest.approx(0.28)
    assert buffer.add_pcm(
        "first",
        pcm_byte_count=14_250,
        sample_rate=24_000,
        num_channels=1,
    ) == ()
    now[0] = 10.98
    assert buffer.deadline_due()
    assert buffer.release_deadline() == ("first",)
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["deadline_guard_ms"] == 20.0
    assert telemetry["first_audio_deadline_slack_ms"] == pytest.approx(300.0)


def test_playback_deadline_guard_requires_deadline() -> None:
    with pytest.raises(ValueError, match="requires"):
        playback_start_config_from_headers(
            {
                "x-vllm-omni-playback-buffer-ms": "321",
                PLAYBACK_DEADLINE_GUARD_MS_HEADER: "20",
            },
            request_start_s=0.0,
            trusted=True,
        )


def test_deadline_without_target_does_not_enable_playback_buffer() -> None:
    config = playback_start_config_from_headers(
        {"x-vllm-omni-first-output-deadline-ms": "750"},
        request_start_s=10.0,
        trusted=True,
    )
    assert config is None


def test_admission_and_playback_share_ingress_deadline_after_preprocessing() -> None:
    headers = {
        "x-vllm-omni-playback-buffer-ms": "300",
        "x-vllm-omni-first-output-deadline-ms": "900",
    }
    playback = playback_start_config_from_headers(
        headers,
        request_start_s=10.0,
        trusted=True,
    )
    scheduling = scheduling_kwargs_from_headers(
        headers,
        trusted=True,
        deadline_anchor_monotonic_s=10.0,
    )
    metadata = RequestSchedulingMetadata.create(
        **scheduling,
        now_monotonic_s=10.6,
    )

    assert playback is not None
    assert playback.deadline_monotonic_s == 10.9
    assert metadata.deadline_monotonic_s == playback.deadline_monotonic_s


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
            pcm_byte_count=200,
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
        pcm_byte_count=400,
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
        "deadline_guard_ms": 0.0,
        "first_audio_deadline_slack_ms": None,
    }


def test_eos_flushes_short_audio_and_error_discards_it() -> None:
    buffer = PlaybackStartBuffer(PlaybackStartConfig(target_ms=500.0), clock=lambda: 1.0)
    assert buffer.add_pcm("pcm", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ()
    assert buffer.finish() == ("pcm",)
    assert buffer.telemetry(status="ok")["release_reason"] == "eos"

    failed = PlaybackStartBuffer(PlaybackStartConfig(target_ms=500.0), clock=lambda: 1.0)
    assert failed.add_pcm("pcm", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ()
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
    pending_delivery = asyncio.ensure_future(anext(stream))
    await asyncio.wait_for(engine_waiting.wait(), timeout=1.0)
    pending_delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_delivery
    assert generator_closed.is_set()
