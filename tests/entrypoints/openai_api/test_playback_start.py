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
    PLAYBACK_DEADLINE_EVENT,
    PLAYBACK_DEADLINE_GUARD_MS_HEADER,
    PLAYBACK_DEADLINE_MIN_BUFFER_MS_HEADER,
    PLAYBACK_RELEASE_MODE_HEADER,
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
        deadline_guard_min_buffer_ms=None,
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
    assert config.deadline_guard_min_buffer_ms is None

    now = [10.7]
    buffer = PlaybackStartBuffer(config, clock=lambda: now[0])
    assert buffer.seconds_until_deadline() == pytest.approx(0.28)
    assert (
        buffer.add_pcm(
            "first",
            pcm_byte_count=14_250,
            sample_rate=24_000,
            num_channels=1,
        )
        == ()
    )
    now[0] = 10.98
    assert buffer.deadline_due()
    assert buffer.release_deadline() == ("first",)
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["deadline_guard_ms"] == 20.0
    assert telemetry["deadline_guard_min_buffer_ms"] is None
    assert telemetry["deadline_guard_evaluated"] is False
    assert telemetry["deadline_guard_deferred"] is False
    assert telemetry["deadline_guard_released"] is False
    assert telemetry["first_audio_deadline_slack_ms"] == pytest.approx(300.0)


def test_selective_deadline_guard_defers_until_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    config = playback_start_config_from_headers(
        {
            "x-vllm-omni-playback-buffer-ms": "321",
            "x-vllm-omni-first-output-deadline-ms": "1000",
            PLAYBACK_DEADLINE_GUARD_MS_HEADER: "20",
            PLAYBACK_DEADLINE_MIN_BUFFER_MS_HEADER: "120",
        },
        request_start_s=10.0,
    )
    assert config == PlaybackStartConfig(
        target_ms=321.0,
        deadline_monotonic_s=11.0,
        deadline_guard_ms=20.0,
        deadline_guard_min_buffer_ms=120.0,
    )

    now = [10.98]
    buffer = PlaybackStartBuffer(config, clock=lambda: now[0])
    assert buffer.deadline_due()
    assert buffer.release_deadline() == ()
    assert not buffer.released
    assert buffer.seconds_until_deadline() == pytest.approx(0.02)
    assert (
        buffer.add_pcm(
            "first",
            pcm_byte_count=4_800,
            sample_rate=24_000,
            num_channels=1,
        )
        == ()
    )
    now[0] = 11.0
    assert buffer.deadline_due()
    assert buffer.release_deadline() == ("first",)
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["release_reason"] == "deadline"
    assert telemetry["deadline_guard_evaluated"] is True
    assert telemetry["deadline_guard_deferred"] is True
    assert telemetry["deadline_guard_released"] is False


def test_selective_deadline_guard_releases_sufficient_buffer() -> None:
    now = [10.9]
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(
            target_ms=321.0,
            deadline_monotonic_s=11.0,
            deadline_guard_ms=20.0,
            deadline_guard_min_buffer_ms=120.0,
        ),
        clock=lambda: now[0],
    )
    assert (
        buffer.add_pcm(
            "first",
            pcm_byte_count=14_250,
            sample_rate=24_000,
            num_channels=1,
        )
        == ()
    )
    now[0] = 10.98
    assert buffer.release_deadline() == ("first",)
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["release_reason"] == "deadline_guard"
    assert telemetry["deadline_guard_evaluated"] is True
    assert telemetry["deadline_guard_deferred"] is False
    assert telemetry["deadline_guard_released"] is True


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


def test_playback_deadline_min_buffer_requires_guard() -> None:
    with pytest.raises(ValueError, match="requires"):
        playback_start_config_from_headers(
            {
                "x-vllm-omni-playback-buffer-ms": "321",
                "x-vllm-omni-first-output-deadline-ms": "1000",
                PLAYBACK_DEADLINE_MIN_BUFFER_MS_HEADER: "120",
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
        "deadline_guard_min_buffer_ms": None,
        "deadline_guard_evaluated": False,
        "deadline_guard_deferred": False,
        "deadline_guard_released": False,
        "first_audio_deadline_slack_ms": None,
        "release_mode": "target",
        "buffer_limit_released": False,
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


def _deadline_headers(**overrides: str) -> dict[str, str]:
    return {
        PLAYBACK_RELEASE_MODE_HEADER: "deadline",
        "x-vllm-omni-first-output-deadline-ms": "1000",
        PLAYBACK_DEADLINE_GUARD_MS_HEADER: "20",
        **overrides,
    }


def test_explicit_deadline_mode_requires_trust_and_uses_fixed_guard() -> None:
    headers = _deadline_headers()
    assert playback_start_config_from_headers(headers, request_start_s=10.0, trusted=False) is None
    config = playback_start_config_from_headers(headers, request_start_s=10.0, trusted=True)
    assert config == PlaybackStartConfig(
        target_ms=MAX_PLAYBACK_BUFFER_MS,
        deadline_monotonic_s=11.0,
        deadline_guard_ms=20.0,
        release_mode="deadline",
    )
    buffer = PlaybackStartBuffer(config, clock=lambda: 10.9)
    assert buffer.seconds_until_deadline() == pytest.approx(0.08)
    assert buffer.add_pcm("first", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ()
    assert buffer.release_deadline(now=10.98) == ("first",)
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["release_mode"] == "deadline"
    assert telemetry["release_reason"] == "deadline"
    assert telemetry["deadline_fallback"] is False
    assert telemetry["buffer_limit_released"] is False


@pytest.mark.parametrize("missing", ["x-vllm-omni-first-output-deadline-ms", PLAYBACK_DEADLINE_GUARD_MS_HEADER])
def test_deadline_mode_requires_deadline_and_explicit_guard(missing: str) -> None:
    headers = _deadline_headers()
    del headers[missing]
    with pytest.raises(ValueError, match="requires"):
        playback_start_config_from_headers(headers, request_start_s=0.0, trusted=True)


@pytest.mark.parametrize("conflict", ["x-vllm-omni-playback-buffer-ms", PLAYBACK_DEADLINE_MIN_BUFFER_MS_HEADER])
def test_deadline_mode_rejects_target_and_conditional_guard(conflict: str) -> None:
    with pytest.raises(ValueError, match="does not allow"):
        playback_start_config_from_headers(_deadline_headers(**{conflict: "0"}), request_start_s=0.0, trusted=True)


@pytest.mark.parametrize("guard", ["-1", "nan", "inf", "1001"])
def test_deadline_mode_rejects_invalid_or_excessive_guard(guard: str) -> None:
    with pytest.raises(ValueError, match="playback-deadline-guard-ms"):
        playback_start_config_from_headers(
            _deadline_headers(**{PLAYBACK_DEADLINE_GUARD_MS_HEADER: guard}), request_start_s=0.0, trusted=True
        )


@pytest.mark.parametrize("guard", ["0", "1000"])
def test_deadline_mode_accepts_guard_budget_boundaries(guard: str) -> None:
    config = playback_start_config_from_headers(
        _deadline_headers(**{PLAYBACK_DEADLINE_GUARD_MS_HEADER: guard}), request_start_s=10.0, trusted=True
    )
    assert config is not None
    assert PlaybackStartBuffer(config).deadline_monotonic_s == pytest.approx(11.0 - float(guard) / 1000.0)


def test_invalid_release_mode_is_rejected_and_explicit_target_preserves_defaults() -> None:
    with pytest.raises(ValueError, match="release-mode"):
        playback_start_config_from_headers({PLAYBACK_RELEASE_MODE_HEADER: "invalid"}, request_start_s=0.0, trusted=True)
    config = playback_start_config_from_headers(
        {PLAYBACK_RELEASE_MODE_HEADER: "target", "x-vllm-omni-playback-buffer-ms": "321"},
        request_start_s=0.0,
        trusted=True,
    )
    assert config == PlaybackStartConfig(target_ms=321.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"release_mode": "invalid"},
        {"deadline_monotonic_s": None},
        {"deadline_monotonic_s": float("inf")},
        {"deadline_guard_min_buffer_ms": 0.0},
        {"target_ms": 321.0},
        {"deadline_guard_ms": -1.0},
    ],
)
def test_direct_deadline_config_rejects_incompatible_inputs(overrides: dict) -> None:
    params = {"target_ms": MAX_PLAYBACK_BUFFER_MS, "deadline_monotonic_s": 100.0, "release_mode": "deadline"}
    with pytest.raises(ValueError):
        PlaybackStartConfig(**(params | overrides))


@pytest.mark.parametrize("pcm_frames", [60_000, 60_001])
def test_deadline_mode_retention_limit_has_distinct_release_reason(pcm_frames: int) -> None:
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(target_ms=MAX_PLAYBACK_BUFFER_MS, deadline_monotonic_s=100.0, release_mode="deadline"),
        clock=lambda: 1.0,
    )
    assert buffer.add_pcm("empty", pcm_byte_count=0, sample_rate=1000, num_channels=1) == ()
    assert buffer.add_pcm("large", pcm_byte_count=2 * pcm_frames, sample_rate=1000, num_channels=1) == (
        "empty",
        "large",
    )
    telemetry = buffer.telemetry(status="ok")
    assert telemetry["release_reason"] == "buffer_limit"
    assert telemetry["buffer_limit_released"] is True
    assert telemetry["deadline_fallback"] is False
    assert buffer.add_pcm("later", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ("later",)
    assert buffer.finish() == ()


def test_deadline_mode_timer_before_first_audio_adds_no_late_hold() -> None:
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(target_ms=MAX_PLAYBACK_BUFFER_MS, deadline_monotonic_s=1.0, release_mode="deadline"),
        clock=lambda: 1.0,
    )
    assert buffer.deadline_due()
    assert buffer.release_deadline() == ()
    assert buffer.released
    assert buffer.seconds_until_deadline() is None
    assert buffer.add_pcm("empty", pcm_byte_count=0, sample_rate=1000, num_channels=1) == ("empty",)
    assert buffer.add_pcm("late", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ("late",)
    assert buffer.telemetry(status="ok")["hold_ms"] == 0.0


@pytest.mark.parametrize("terminal", ["eos", "error", "cancelled"])
def test_deadline_mode_eos_flushes_but_failures_discard_held_audio(terminal: str) -> None:
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(target_ms=MAX_PLAYBACK_BUFFER_MS, deadline_monotonic_s=100.0, release_mode="deadline"),
        clock=lambda: 1.0,
    )
    assert buffer.add_pcm("audio", pcm_byte_count=200, sample_rate=1000, num_channels=1) == ()
    if terminal == "eos":
        assert buffer.finish() == ("audio",)
    else:
        buffer.terminate(terminal)
        assert buffer.finish() == ()
    assert buffer.telemetry(status="ok" if terminal == "eos" else terminal)["release_reason"] == terminal


@pytest.mark.asyncio
async def test_explicit_deadline_mode_keeps_pending_engine_pull_alive() -> None:
    engine_waiting = asyncio.Event()
    allow_second_audio = asyncio.Event()
    engine_cancelled = asyncio.Event()

    async def results():
        yield "first"
        engine_waiting.set()
        try:
            await allow_second_audio.wait()
        except asyncio.CancelledError:
            engine_cancelled.set()
            raise
        yield "second"

    loop = asyncio.get_running_loop()
    buffer = PlaybackStartBuffer(
        PlaybackStartConfig(
            target_ms=MAX_PLAYBACK_BUFFER_MS,
            deadline_monotonic_s=loop.time() + 0.02,
            deadline_guard_ms=10.0,
            release_mode="deadline",
        ),
        clock=loop.time,
    )
    stream = iterate_with_playback_deadline(results(), buffer)
    first = await anext(stream)
    assert buffer.add_pcm(first, pcm_byte_count=200, sample_rate=1000, num_channels=1) == ()
    assert await asyncio.wait_for(anext(stream), timeout=1.0) is PLAYBACK_DEADLINE_EVENT
    assert engine_waiting.is_set()
    assert not engine_cancelled.is_set()
    assert buffer.release_deadline() == ("first",)
    allow_second_audio.set()
    assert await asyncio.wait_for(anext(stream), timeout=1.0) == "second"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not engine_cancelled.is_set()
