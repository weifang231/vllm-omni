# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Coordinator-owned WIP credits and request ordering for Omni pipelines.

This module is intentionally independent of CUDA and vLLM scheduler internals.
The orchestrator owns one instance and admits work to stage engines only after
the corresponding pipeline, path, class, and stage credits are available.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

QueuePolicy = Literal["fifo", "edf"]
DispatchCallable = Callable[[], Awaitable[bool]]

REQUEST_CLASS_HEADER = "x-vllm-omni-request-class"
REQUEST_PATH_HEADER = "x-vllm-omni-request-path"
FIRST_OUTPUT_DEADLINE_MS_HEADER = "x-vllm-omni-first-output-deadline-ms"
TRUST_SCHEDULING_HEADERS_ENV = "VLLM_OMNI_TRUST_SCHEDULING_HEADERS"


def _label(value: str | None, *, default: str, field_name: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if len(normalized) > 128 or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must be at most 128 printable characters")
    return normalized


def scheduling_kwargs_from_headers(
    headers: Mapping[str, str] | None,
    *,
    trusted: bool | None = None,
) -> dict[str, Any]:
    """Translate trusted HTTP headers into :meth:`AsyncOmni.generate` kwargs.

    Headers are ignored by default because accepting caller-selected classes or
    deadlines at a public ingress would let clients evade class limits or gain
    EDF priority. Operators may opt in only when a trusted proxy owns these
    headers, or callers may pass ``trusted=True`` at an internal boundary.
    """
    if trusted is None:
        trusted = os.environ.get(TRUST_SCHEDULING_HEADERS_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not trusted or headers is None:
        return {}
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    kwargs: dict[str, Any] = {}
    if REQUEST_CLASS_HEADER in normalized_headers:
        request_class = normalized_headers[REQUEST_CLASS_HEADER]
        kwargs["request_class"] = _label(
            request_class,
            default="default",
            field_name=REQUEST_CLASS_HEADER,
        )
    if REQUEST_PATH_HEADER in normalized_headers:
        path = normalized_headers[REQUEST_PATH_HEADER]
        kwargs["request_path"] = _label(
            path,
            default="default",
            field_name=REQUEST_PATH_HEADER,
        )
    if FIRST_OUTPUT_DEADLINE_MS_HEADER in normalized_headers:
        raw_deadline_ms = normalized_headers[FIRST_OUTPUT_DEADLINE_MS_HEADER]
        try:
            deadline_ms = float(raw_deadline_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FIRST_OUTPUT_DEADLINE_MS_HEADER} must be a number") from exc
        if not math.isfinite(deadline_ms) or deadline_ms < 0:
            raise ValueError(f"{FIRST_OUTPUT_DEADLINE_MS_HEADER} must be finite and non-negative")
        kwargs["first_output_deadline_s"] = deadline_ms / 1000.0
    return kwargs


def _optional_limit(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer or null")
    return value


def _limit_map(
    value: Any,
    *,
    field_name: str,
    key_parser: Callable[[Any], Any],
) -> dict[Any, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    parsed: dict[Any, int] = {}
    for raw_key, raw_limit in value.items():
        key = key_parser(raw_key)
        limit = _optional_limit(raw_limit, field_name=f"{field_name}[{raw_key!r}]")
        if limit is None:
            raise ValueError(f"{field_name}[{raw_key!r}] must not be null")
        parsed[key] = limit
    return parsed


def _stage_key(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("stage limit keys must be non-negative integers")
    try:
        stage_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid stage id {value!r}") from exc
    if stage_id < 0 or str(stage_id) != str(value).strip():
        raise ValueError(f"invalid stage id {value!r}")
    return stage_id


def _string_key(value: Any) -> str:
    return _label(str(value), default="default", field_name="limit key")


@dataclass(frozen=True, slots=True)
class RequestSchedulingMetadata:
    """Observable metadata used by the coordinator queue.

    ``deadline_monotonic_s`` is process-local and absolute on
    :func:`time.monotonic`'s clock. It is never interpreted as wall time.
    """

    request_class: str = "default"
    path: str = "default"
    deadline_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_class",
            _label(self.request_class, default="default", field_name="request_class"),
        )
        object.__setattr__(
            self,
            "path",
            _label(self.path, default="default", field_name="path"),
        )
        if self.deadline_monotonic_s is not None:
            deadline = float(self.deadline_monotonic_s)
            if not math.isfinite(deadline):
                raise ValueError("deadline_monotonic_s must be finite")
            object.__setattr__(self, "deadline_monotonic_s", deadline)

    @classmethod
    def create(
        cls,
        *,
        request_class: str | None = None,
        path: str | None = None,
        default_path: str = "default",
        first_output_deadline_s: float | None = None,
        now_monotonic_s: float | None = None,
    ) -> RequestSchedulingMetadata:
        normalized_path = _label(path, default=default_path, field_name="path")
        normalized_class = _label(
            request_class,
            default=normalized_path,
            field_name="request_class",
        )
        deadline = None
        if first_output_deadline_s is not None:
            budget = float(first_output_deadline_s)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError("first_output_deadline_s must be finite and non-negative")
            now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
            deadline = now + budget
        return cls(
            request_class=normalized_class,
            path=normalized_path,
            deadline_monotonic_s=deadline,
        )


@dataclass(frozen=True, slots=True)
class QueueControlConfig:
    """Validated queue-control settings loaded from ``queue_control`` JSON."""

    enabled: bool = False
    policy: QueuePolicy = "fifo"
    global_wip_limit: int | None = None
    stage_wip_limits: dict[int, int] = field(default_factory=dict)
    path_wip_limits: dict[str, int] = field(default_factory=dict)
    class_wip_limits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        num_stages: int | None = None,
    ) -> QueueControlConfig:
        raw = document.get("queue_control")
        if raw is None or raw is False:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("queue_control must be a JSON object or false")

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("queue_control.enabled must be boolean")
        policy = str(raw.get("policy", "fifo")).strip().lower()
        if policy not in {"fifo", "edf"}:
            raise ValueError("queue_control.policy must be 'fifo' or 'edf'")

        stage_limits = _limit_map(
            raw.get("stage_wip_limits"),
            field_name="queue_control.stage_wip_limits",
            key_parser=_stage_key,
        )
        if num_stages is not None:
            invalid = sorted(stage_id for stage_id in stage_limits if stage_id >= num_stages)
            if invalid:
                raise ValueError(f"stage_wip_limits contains unavailable stage ids: {invalid}")

        return cls(
            enabled=enabled,
            policy=policy,  # type: ignore[arg-type]
            global_wip_limit=_optional_limit(
                raw.get("global_wip_limit"),
                field_name="queue_control.global_wip_limit",
            ),
            stage_wip_limits=stage_limits,
            path_wip_limits=_limit_map(
                raw.get("path_wip_limits"),
                field_name="queue_control.path_wip_limits",
                key_parser=_string_key,
            ),
            class_wip_limits=_limit_map(
                raw.get("class_wip_limits"),
                field_name="queue_control.class_wip_limits",
                key_parser=_string_key,
            ),
        )


@dataclass(slots=True)
class PendingStageDispatch:
    request_id: str
    logical_request_id: str
    stage_id: int
    metadata: RequestSchedulingMetadata
    dispatch: DispatchCallable
    operation: str
    starts_request: bool = False
    required_active_stage_id: int | None = None
    sequence: int = 0
    enqueued_monotonic_s: float = 0.0


@dataclass(frozen=True, slots=True)
class AcquiredStageDispatch:
    pending: PendingStageDispatch
    acquired_request: bool
    acquired_stage: bool
    queue_wait_s: float


class RuntimeQueueController:
    """Bookkeep queue order and non-preemptive WIP-credit leases."""

    def __init__(
        self,
        *,
        num_stages: int,
        config: QueueControlConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if num_stages < 0:
            raise ValueError("num_stages must be non-negative")
        self.num_stages = num_stages
        self.config = config or QueueControlConfig()
        self._clock = clock
        self._pending: list[PendingStageDispatch] = []
        self._next_sequence = 0
        self._active_requests: dict[str, RequestSchedulingMetadata] = {}
        self._request_to_logical: dict[str, str] = {}
        self._active_stages: set[tuple[str, int]] = set()
        self._config_generation = 0
        self._enqueued_total = 0
        self._dispatch_attempts_total = 0
        self._dispatch_failures_total = 0
        self._cancelled_total = 0
        self._queue_wait_s_total = 0.0
        self._queue_wait_s_max = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def configure(self, config: QueueControlConfig) -> bool:
        if config == self.config:
            return False
        self.config = config
        self._config_generation += 1
        return True

    def enqueue(self, pending: PendingStageDispatch) -> None:
        if pending.stage_id < 0 or pending.stage_id >= self.num_stages:
            raise ValueError(f"stage_id {pending.stage_id} is outside [0, {self.num_stages})")
        pending.sequence = self._next_sequence
        self._next_sequence += 1
        pending.enqueued_monotonic_s = self._clock()
        self._pending.append(pending)
        self._enqueued_total += 1

    def _request_counts(self) -> tuple[Counter[str], Counter[str]]:
        path_counts = Counter(metadata.path for metadata in self._active_requests.values())
        class_counts = Counter(metadata.request_class for metadata in self._active_requests.values())
        return path_counts, class_counts

    def _blocked_reasons(self, pending: PendingStageDispatch) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        reasons: list[str] = []
        if (
            pending.required_active_stage_id is not None
            and (
                pending.request_id,
                pending.required_active_stage_id,
            )
            not in self._active_stages
        ):
            reasons.append("dependency")
        stage_key = (pending.request_id, pending.stage_id)
        if stage_key not in self._active_stages:
            stage_limit = self.config.stage_wip_limits.get(pending.stage_id)
            if stage_limit is not None:
                active = sum(stage_id == pending.stage_id for _, stage_id in self._active_stages)
                if active >= stage_limit:
                    reasons.append("stage")

        if pending.logical_request_id not in self._active_requests:
            if not pending.starts_request:
                reasons.append("request")
                return tuple(reasons)
            global_limit = self.config.global_wip_limit
            if global_limit is not None and len(self._active_requests) >= global_limit:
                reasons.append("global")
            path_counts, class_counts = self._request_counts()
            path_limit = self.config.path_wip_limits.get(pending.metadata.path)
            if path_limit is not None and path_counts[pending.metadata.path] >= path_limit:
                reasons.append("path")
            class_limit = self.config.class_wip_limits.get(pending.metadata.request_class)
            if class_limit is not None and class_counts[pending.metadata.request_class] >= class_limit:
                reasons.append("class")
        return tuple(reasons)

    def _order_key(self, pending: PendingStageDispatch) -> tuple[float, int]:
        if self.config.policy == "edf":
            deadline = pending.metadata.deadline_monotonic_s
            return (math.inf if deadline is None else deadline, pending.sequence)
        return (float(pending.sequence), pending.sequence)

    def pop_ready(self) -> AcquiredStageDispatch | None:
        if not self._pending:
            return None
        ordered_indices = sorted(range(len(self._pending)), key=lambda index: self._order_key(self._pending[index]))
        selected_index = next(
            (index for index in ordered_indices if not self._blocked_reasons(self._pending[index])),
            None,
        )
        if selected_index is None:
            return None

        pending = self._pending.pop(selected_index)
        acquired_request = pending.logical_request_id not in self._active_requests
        acquired_stage = (pending.request_id, pending.stage_id) not in self._active_stages
        if acquired_request:
            self._active_requests[pending.logical_request_id] = pending.metadata
        self._request_to_logical[pending.request_id] = pending.logical_request_id
        if acquired_stage:
            self._active_stages.add((pending.request_id, pending.stage_id))

        queue_wait_s = max(self._clock() - pending.enqueued_monotonic_s, 0.0)
        self._dispatch_attempts_total += 1
        self._queue_wait_s_total += queue_wait_s
        self._queue_wait_s_max = max(self._queue_wait_s_max, queue_wait_s)
        return AcquiredStageDispatch(
            pending=pending,
            acquired_request=acquired_request,
            acquired_stage=acquired_stage,
            queue_wait_s=queue_wait_s,
        )

    def rollback(self, acquired: AcquiredStageDispatch) -> None:
        pending = acquired.pending
        if acquired.acquired_stage:
            self._active_stages.discard((pending.request_id, pending.stage_id))
        if acquired.acquired_request:
            self._active_requests.pop(pending.logical_request_id, None)
        if not any(request_id == pending.request_id for request_id, _ in self._active_stages):
            self._request_to_logical.pop(pending.request_id, None)
        self._dispatch_failures_total += 1

    def release_stage(self, request_id: str, stage_id: int) -> bool:
        key = (request_id, stage_id)
        if key not in self._active_stages:
            return False
        self._active_stages.remove(key)
        return True

    def cancel_request(self, request_id: str) -> None:
        logical_id = self._request_to_logical.get(request_id, request_id)
        before = len(self._pending)
        if request_id == logical_id:
            self._pending = [item for item in self._pending if item.logical_request_id != logical_id]
            self._active_stages = {
                key for key in self._active_stages if self._request_to_logical.get(key[0], key[0]) != logical_id
            }
            self._active_requests.pop(logical_id, None)
            for actual_id, mapped_logical in list(self._request_to_logical.items()):
                if mapped_logical == logical_id:
                    self._request_to_logical.pop(actual_id, None)
        else:
            self._pending = [item for item in self._pending if item.request_id != request_id]
            self._active_stages = {key for key in self._active_stages if key[0] != request_id}
            self._request_to_logical.pop(request_id, None)
        self._cancelled_total += before - len(self._pending)

    def snapshot(self) -> dict[str, Any]:
        path_counts, class_counts = self._request_counts()
        stage_counts = Counter(stage_id for _, stage_id in self._active_stages)
        queued_stage_counts = Counter(item.stage_id for item in self._pending)
        blocked = Counter(reason for pending in self._pending for reason in self._blocked_reasons(pending))
        now = self._clock()
        return {
            "enabled": self.enabled,
            "policy": self.config.policy,
            "config_generation": self._config_generation,
            "global_wip_limit": self.config.global_wip_limit,
            "stage_wip_limits": {str(key): value for key, value in sorted(self.config.stage_wip_limits.items())},
            "path_wip_limits": dict(sorted(self.config.path_wip_limits.items())),
            "class_wip_limits": dict(sorted(self.config.class_wip_limits.items())),
            "active_requests": len(self._active_requests),
            "active_by_stage": {str(key): value for key, value in sorted(stage_counts.items())},
            "active_by_path": dict(sorted(path_counts.items())),
            "active_by_class": dict(sorted(class_counts.items())),
            "queued_requests": len(self._pending),
            "queued_by_stage": {str(key): value for key, value in sorted(queued_stage_counts.items())},
            "blocked_by_limit": dict(sorted(blocked.items())),
            "oldest_queue_wait_s": max(
                (max(now - pending.enqueued_monotonic_s, 0.0) for pending in self._pending),
                default=0.0,
            ),
            "enqueued_total": self._enqueued_total,
            "dispatch_attempts_total": self._dispatch_attempts_total,
            "dispatch_failures_total": self._dispatch_failures_total,
            "cancelled_total": self._cancelled_total,
            "queue_wait_s_total": self._queue_wait_s_total,
            "queue_wait_s_max": self._queue_wait_s_max,
        }
