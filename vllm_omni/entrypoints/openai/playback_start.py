# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in playback-start buffering for streaming audio responses.

The controller chooses a startup-buffer target.  This module only implements
the serving mechanism that holds playable PCM until that target, the
first-output deadline, or a clean end of stream.  It never replays an already
delivered chunk and does not stop the serving loop from draining the engine.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.logger import init_logger

from vllm_omni.engine.queue_control import (
    FIRST_OUTPUT_DEADLINE_MS_HEADER,
    scheduling_headers_trusted,
)

PLAYBACK_BUFFER_MS_HEADER = "x-vllm-omni-playback-buffer-ms"
PLAYBACK_DEADLINE_GUARD_MS_HEADER = "x-vllm-omni-playback-deadline-guard-ms"

# A trusted proxy can select the target, but a typo must not make the adapter
# retain an arbitrarily long response.  At 24 kHz mono PCM16 this limit is
# about 2.75 MiB, plus at most one runtime audio chunk.
MAX_PLAYBACK_BUFFER_MS = 60_000.0

PlaybackReleaseReason = Literal["target", "deadline", "eos", "cancelled", "error"]
PlaybackTerminalStatus = Literal["ok", "cancelled", "engine_dead", "error"]
PLAYBACK_DEADLINE_EVENT = object()

logger = init_logger(__name__)


def _nonnegative_header_ms(value: Any, *, field_name: str, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be at most {maximum:g}")
    return parsed


@dataclass(frozen=True, slots=True)
class PlaybackStartConfig:
    """Per-request mechanism inputs supplied by a trusted ingress."""

    target_ms: float
    deadline_monotonic_s: float | None = None
    deadline_guard_ms: float = 0.0


def playback_start_config_from_headers(
    headers: Mapping[str, str] | None,
    *,
    request_start_s: float,
    trusted: bool | None = None,
) -> PlaybackStartConfig | None:
    """Parse the opt-in playback target and its deadline fallback.

    The same trust gate as queue scheduling metadata applies.  A deadline by
    itself does not enable buffering: the controller must explicitly provide a
    startup-buffer target.
    """
    if not scheduling_headers_trusted(trusted=trusted) or headers is None:
        return None
    normalized = {str(key).lower(): value for key, value in headers.items()}
    raw_target = normalized.get(PLAYBACK_BUFFER_MS_HEADER)
    if raw_target is None:
        return None
    target_ms = _nonnegative_header_ms(
        raw_target,
        field_name=PLAYBACK_BUFFER_MS_HEADER,
        maximum=MAX_PLAYBACK_BUFFER_MS,
    )
    deadline_monotonic_s = None
    deadline_guard_ms = 0.0
    if FIRST_OUTPUT_DEADLINE_MS_HEADER in normalized:
        deadline_ms = _nonnegative_header_ms(
            normalized[FIRST_OUTPUT_DEADLINE_MS_HEADER],
            field_name=FIRST_OUTPUT_DEADLINE_MS_HEADER,
        )
        deadline_monotonic_s = request_start_s + deadline_ms / 1000.0
    raw_guard = normalized.get(PLAYBACK_DEADLINE_GUARD_MS_HEADER)
    if raw_guard is not None:
        if deadline_monotonic_s is None:
            raise ValueError(
                f"{PLAYBACK_DEADLINE_GUARD_MS_HEADER} requires "
                f"{FIRST_OUTPUT_DEADLINE_MS_HEADER}"
            )
        deadline_guard_ms = _nonnegative_header_ms(
            raw_guard,
            field_name=PLAYBACK_DEADLINE_GUARD_MS_HEADER,
            maximum=MAX_PLAYBACK_BUFFER_MS,
        )
    return PlaybackStartConfig(
        target_ms=target_ms,
        deadline_monotonic_s=deadline_monotonic_s,
        deadline_guard_ms=deadline_guard_ms,
    )


@dataclass(slots=True)
class PlaybackStartBuffer:
    """Hold ordered delivery items until the request's playback gate opens."""

    config: PlaybackStartConfig
    clock: Callable[[], float] = time.monotonic
    _pending: list[Any] = field(default_factory=list)
    _buffered_audio_ms: float = 0.0
    _first_audio_ready_s: float | None = None
    _first_audio_deadline_slack_ms: float | None = None
    _released_s: float | None = None
    _release_reason: PlaybackReleaseReason | None = None
    _telemetry_recorded: bool = False

    @property
    def released(self) -> bool:
        return self._released_s is not None

    @property
    def deadline_monotonic_s(self) -> float | None:
        if self.config.deadline_monotonic_s is None:
            return None
        return self.config.deadline_monotonic_s - self.config.deadline_guard_ms / 1000.0

    def seconds_until_deadline(self, *, now: float | None = None) -> float | None:
        if self.released or self.deadline_monotonic_s is None:
            return None
        current = self.clock() if now is None else now
        return max(self.deadline_monotonic_s - current, 0.0)

    def deadline_due(self, *, now: float | None = None) -> bool:
        remaining = self.seconds_until_deadline(now=now)
        return remaining is not None and remaining <= 0.0

    def add_pcm(
        self,
        delivery_item: Any,
        *,
        pcm_byte_count: int,
        sample_rate: int,
        num_channels: int,
        prefix_items: tuple[Any, ...] = (),
        now: float | None = None,
    ) -> tuple[Any, ...]:
        """Add one PCM16 chunk and return items released by this event."""
        if sample_rate <= 0:
            raise ValueError("PCM sample rate must be positive")
        if num_channels <= 0:
            raise ValueError("PCM channel count must be positive")
        frame_width = 2 * num_channels
        if pcm_byte_count < 0 or pcm_byte_count % frame_width:
            raise ValueError("PCM16 byte count must align to complete audio frames")

        current = self.clock() if now is None else now
        if self.released:
            return (*prefix_items, delivery_item)
        if pcm_byte_count > 0 and self._first_audio_ready_s is None:
            self._first_audio_ready_s = current
            if self.config.deadline_monotonic_s is not None:
                self._first_audio_deadline_slack_ms = (
                    self.config.deadline_monotonic_s - current
                ) * 1000.0
        self._pending.extend(prefix_items)
        self._pending.append(delivery_item)
        frames = pcm_byte_count // frame_width
        self._buffered_audio_ms += frames * 1000.0 / sample_rate
        if self._buffered_audio_ms >= self.config.target_ms:
            return self.release("target", now=current)
        return ()

    def release_deadline(self, *, now: float | None = None) -> tuple[Any, ...]:
        return self.release("deadline", now=now)

    def finish(self, *, now: float | None = None) -> tuple[Any, ...]:
        """Flush a short final utterance that never reached its target."""
        return self.release("eos", now=now)

    def terminate(self, reason: Literal["cancelled", "error"], *, now: float | None = None) -> None:
        """Record a terminal failure without turning held bytes into output."""
        if self.released:
            return
        current = self.clock() if now is None else now
        self._released_s = current
        self._release_reason = reason
        self._pending.clear()

    def release(
        self,
        reason: Literal["target", "deadline", "eos"],
        *,
        now: float | None = None,
    ) -> tuple[Any, ...]:
        if self.released:
            return ()
        self._released_s = self.clock() if now is None else now
        self._release_reason = reason
        pending = tuple(self._pending)
        self._pending.clear()
        return pending

    def telemetry(self, *, status: PlaybackTerminalStatus) -> dict[str, Any]:
        """Return one bounded-cardinality record for the completed stream."""
        hold_ms = 0.0
        if self._first_audio_ready_s is not None and self._released_s is not None:
            hold_ms = max((self._released_s - self._first_audio_ready_s) * 1000.0, 0.0)
        return {
            "status": status,
            "target_ms": round(self.config.target_ms, 3),
            "buffered_audio_ms": round(self._buffered_audio_ms, 3),
            "hold_ms": round(hold_ms, 3),
            "release_reason": self._release_reason or "none",
            "deadline_fallback": self._release_reason == "deadline",
            "deadline_guard_ms": round(self.config.deadline_guard_ms, 3),
            "first_audio_deadline_slack_ms": (
                None
                if self._first_audio_deadline_slack_ms is None
                else round(self._first_audio_deadline_slack_ms, 3)
            ),
        }

    def record_telemetry(
        self,
        *,
        request_id: str,
        status: PlaybackTerminalStatus,
        request_state: Any | None = None,
    ) -> dict[str, Any] | None:
        """Store and log at most one terminal record for this request."""
        if self._telemetry_recorded:
            return None
        self._telemetry_recorded = True
        telemetry = self.telemetry(status=status)
        if request_state is not None:
            request_state.playback_start_telemetry = telemetry
        logger.info(
            "[PlaybackStart] request_id=%s status=%s target_ms=%.3f "
            "buffered_audio_ms=%.3f hold_ms=%.3f release_reason=%s "
            "deadline_fallback=%s deadline_guard_ms=%.3f "
            "first_audio_deadline_slack_ms=%s",
            request_id,
            telemetry["status"],
            telemetry["target_ms"],
            telemetry["buffered_audio_ms"],
            telemetry["hold_ms"],
            telemetry["release_reason"],
            telemetry["deadline_fallback"],
            telemetry["deadline_guard_ms"],
            telemetry["first_audio_deadline_slack_ms"],
        )
        return telemetry


async def iterate_with_playback_deadline(
    generator: AsyncIterator[Any],
    buffer: PlaybackStartBuffer,
) -> AsyncGenerator[Any, None]:
    """Yield engine results and one event when the fallback deadline expires.

    A pending ``__anext__`` call remains active at the deadline so opening the
    playback gate never cancels or restarts engine generation.  If the HTTP
    stream itself is cancelled, the pending pull is cancelled and awaited to
    propagate cancellation into the engine generator.
    """
    iterator = aiter(generator)
    pending_next: asyncio.Future | None = None
    try:
        while True:
            remaining = buffer.seconds_until_deadline()
            if remaining is None:
                if pending_next is None:
                    yield await anext(iterator)
                else:
                    result = await pending_next
                    pending_next = None
                    yield result
                continue

            if remaining <= 0.0:
                yield PLAYBACK_DEADLINE_EVENT
                continue

            if pending_next is None:
                pending_next = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait((pending_next,), timeout=remaining)
            if not done:
                yield PLAYBACK_DEADLINE_EVENT
                continue
            result = pending_next.result()
            pending_next = None
            yield result
    except StopAsyncIteration:
        return
    finally:
        if pending_next is not None:
            if not pending_next.done():
                pending_next.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                await pending_next
