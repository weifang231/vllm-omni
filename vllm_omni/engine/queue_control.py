# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Coordinator-owned WIP credits and request ordering for Omni pipelines.

This module is intentionally independent of CUDA and vLLM scheduler internals.
The orchestrator owns one instance and admits work to stage engines only after
the corresponding pipeline, path, class, stage, and stage-class credits are
available.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

QueuePolicy = Literal["fifo", "edf"]
DispatchCallable = Callable[[], Awaitable[bool]]
AdmissionScoreMethod = Literal["erlang_empirical"]

REQUEST_CLASS_HEADER = "x-vllm-omni-request-class"
REQUEST_PATH_HEADER = "x-vllm-omni-request-path"
FIRST_OUTPUT_DEADLINE_MS_HEADER = "x-vllm-omni-first-output-deadline-ms"
TRUST_SCHEDULING_HEADERS_ENV = "VLLM_OMNI_TRUST_SCHEDULING_HEADERS"


def scheduling_headers_trusted(*, trusted: bool | None = None) -> bool:
    """Return whether caller-supplied scheduling metadata is trusted."""
    if trusted is not None:
        return trusted
    return os.environ.get(TRUST_SCHEDULING_HEADERS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    if not scheduling_headers_trusted(trusted=trusted) or headers is None:
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


def _stage_class_limit_map(value: Any, *, field_name: str) -> dict[int, dict[str, int]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    parsed: dict[int, dict[str, int]] = {}
    for raw_stage_id, raw_class_limits in value.items():
        stage_id = _stage_key(raw_stage_id)
        parsed[stage_id] = _limit_map(
            raw_class_limits,
            field_name=f"{field_name}[{raw_stage_id!r}]",
            key_parser=_string_key,
        )
    return parsed


def _finite_float(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        qualifier = f" greater than or equal to {minimum:g}" if minimum is not None else ""
        raise ValueError(f"{field_name} must be a finite number{qualifier}")
    return parsed


@dataclass(frozen=True, slots=True)
class AdmissionClassConfig:
    """Calibrated Proposition-2 inputs for one request class.

    ``effective_k`` is the active class concurrency limit, ``mu`` is the
    fitted per-occupied-slot service rate at that limit, and
    ``service_samples_s`` are execution-start-to-valid-first-output samples.
    These are empirical model inputs, not a formal latency guarantee.
    """

    effective_k: int
    mu: float
    service_samples_s: tuple[float, ...]
    gamma: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, field_name: str) -> AdmissionClassConfig:
        effective_k = _optional_limit(raw.get("effective_k"), field_name=f"{field_name}.effective_k")
        if effective_k is None:
            raise ValueError(f"{field_name}.effective_k is required")
        mu = _finite_float(raw.get("mu"), field_name=f"{field_name}.mu", minimum=0.0)
        if effective_k > 0 and mu <= 0:
            raise ValueError(f"{field_name}.mu must be positive when effective_k is positive")
        raw_samples = raw.get("service_samples_s")
        if not isinstance(raw_samples, (list, tuple)) or not raw_samples:
            raise ValueError(f"{field_name}.service_samples_s must be a non-empty JSON array")
        samples = tuple(
            _finite_float(sample, field_name=f"{field_name}.service_samples_s[{index}]", minimum=0.0)
            for index, sample in enumerate(raw_samples)
        )
        gamma = _finite_float(raw.get("gamma"), field_name=f"{field_name}.gamma", minimum=0.0)
        if gamma > 1.0:
            raise ValueError(f"{field_name}.gamma must be at most 1")
        return cls(
            effective_k=effective_k,
            mu=mu,
            service_samples_s=samples,
            gamma=gamma,
        )


@dataclass(frozen=True, slots=True)
class AdmissionControlConfig:
    """Opt-in ingress admission settings nested under ``queue_control``."""

    enabled: bool = False
    score_method: AdmissionScoreMethod = "erlang_empirical"
    classes: dict[str, AdmissionClassConfig] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Any) -> AdmissionControlConfig:
        if raw is None or raw is False:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("queue_control.admission must be a JSON object or false")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("queue_control.admission.enabled must be boolean")
        score_method = str(raw.get("score_method", "erlang_empirical")).strip().lower()
        if score_method != "erlang_empirical":
            raise ValueError("queue_control.admission.score_method must be 'erlang_empirical'")
        raw_classes = raw.get("classes", {})
        if not isinstance(raw_classes, Mapping):
            raise ValueError("queue_control.admission.classes must be a JSON object")
        classes: dict[str, AdmissionClassConfig] = {}
        for raw_class, raw_config in raw_classes.items():
            request_class = _label(
                str(raw_class),
                default="default",
                field_name="queue_control.admission class",
            )
            if not isinstance(raw_config, Mapping):
                raise ValueError(f"queue_control.admission.classes[{raw_class!r}] must be a JSON object")
            classes[request_class] = AdmissionClassConfig.from_mapping(
                raw_config,
                field_name=f"queue_control.admission.classes[{raw_class!r}]",
            )
        if enabled and not classes:
            raise ValueError("queue_control.admission.classes must not be empty when admission is enabled")
        return cls(
            enabled=enabled,
            score_method=score_method,  # type: ignore[arg-type]
            classes=classes,
        )


def erlang_wait_cdf(
    wait_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    required_returns: int,
) -> float:
    """Return the Erlang waiting-time CDF in Eq. (6) of the model.

    The Poisson-tail form is evaluated with log-sum-exp so large rate-time
    products do not overflow. Inputs are expected to have passed config
    validation; defensive checks keep this pure helper safe for direct use.
    """

    if wait_budget_s < 0:
        return 0.0
    if required_returns <= 0:
        return 1.0
    if effective_k <= 0 or mu <= 0:
        return 0.0
    x = effective_k * mu * wait_budget_s
    if x <= 0:
        return 0.0

    last = required_returns - 1
    mode = min(int(math.floor(x)), last)
    max_log_probability = -x + mode * math.log(x) - math.lgamma(mode + 1)
    scaled_sum = 0.0
    log_x = math.log(x)
    for count in range(required_returns):
        log_probability = -x + count * log_x - math.lgamma(count + 1)
        scaled_sum += math.exp(log_probability - max_log_probability)
    poisson_cdf = math.exp(max_log_probability) * scaled_sum
    return min(max(1.0 - poisson_cdf, 0.0), 1.0)


def erlang_empirical_admission_score(
    remaining_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    active_count: int,
    queue_position: int,
    service_samples_s: tuple[float, ...],
) -> float:
    """Compute the Proposition-2 Erlang--empirical convolution score."""

    if remaining_budget_s < 0 or effective_k <= 0 or active_count > effective_k:
        return 0.0
    required_returns = max(active_count + queue_position - effective_k + 1, 0)
    if not service_samples_s:
        raise ValueError("service_samples_s must not be empty")
    return sum(
        erlang_wait_cdf(
            remaining_budget_s - sample,
            effective_k=effective_k,
            mu=mu,
            required_returns=required_returns,
        )
        for sample in service_samples_s
    ) / len(service_samples_s)


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
class OnlineAllocatorMetadata:
    """Versioned provenance for one externally selected class allocation.

    The runtime treats this metadata and all queue-control fields as one
    immutable configuration object.  ``revision`` is monotonically increasing
    for cooperating writers, while the source fields identify the causal
    runtime snapshot from which the allocation was computed.
    """

    revision: int
    source_runtime_id: str
    source_snapshot_sequence: int
    source_config_generation: int
    profile_fingerprint: str

    @classmethod
    def from_mapping(cls, raw: Any) -> OnlineAllocatorMetadata | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("queue_control.online_allocator must be a JSON object")
        if raw.get("schema_version") != 1:
            raise ValueError("queue_control.online_allocator.schema_version must be 1")
        revision = _optional_limit(
            raw.get("revision"),
            field_name="queue_control.online_allocator.revision",
        )
        source_snapshot_sequence = _optional_limit(
            raw.get("source_snapshot_sequence"),
            field_name="queue_control.online_allocator.source_snapshot_sequence",
        )
        source_config_generation = _optional_limit(
            raw.get("source_config_generation"),
            field_name="queue_control.online_allocator.source_config_generation",
        )
        if revision is None or revision < 1:
            raise ValueError("queue_control.online_allocator.revision must be at least 1")
        if source_snapshot_sequence is None or source_snapshot_sequence < 1:
            raise ValueError("queue_control.online_allocator.source_snapshot_sequence must be at least 1")
        if source_config_generation is None:
            raise ValueError("queue_control.online_allocator.source_config_generation is required")
        source_runtime_id = _label(
            raw.get("source_runtime_id"),
            default="",
            field_name="queue_control.online_allocator.source_runtime_id",
        )
        if not source_runtime_id:
            raise ValueError("queue_control.online_allocator.source_runtime_id is required")
        profile_fingerprint = _label(
            raw.get("profile_fingerprint"),
            default="",
            field_name="queue_control.online_allocator.profile_fingerprint",
        )
        if len(profile_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in profile_fingerprint
        ):
            raise ValueError(
                "queue_control.online_allocator.profile_fingerprint must be a lowercase SHA-256 hex digest"
            )
        return cls(
            revision=revision,
            source_runtime_id=source_runtime_id,
            source_snapshot_sequence=source_snapshot_sequence,
            source_config_generation=source_config_generation,
            profile_fingerprint=profile_fingerprint,
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": self.revision,
            "source_runtime_id": self.source_runtime_id,
            "source_snapshot_sequence": self.source_snapshot_sequence,
            "source_config_generation": self.source_config_generation,
            "profile_fingerprint": self.profile_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class QueueControlConfig:
    """Validated queue-control settings loaded from ``queue_control`` JSON."""

    enabled: bool = False
    policy: QueuePolicy = "fifo"
    global_wip_limit: int | None = None
    stage_wip_limits: dict[int, int] = field(default_factory=dict)
    stage_class_wip_limits: dict[int, dict[str, int]] = field(default_factory=dict)
    path_wip_limits: dict[str, int] = field(default_factory=dict)
    class_wip_limits: dict[str, int] = field(default_factory=dict)
    admission: AdmissionControlConfig = field(default_factory=AdmissionControlConfig)
    online_allocator: OnlineAllocatorMetadata | None = None

    def __post_init__(self) -> None:
        if self.admission.enabled and self.policy != "edf":
            raise ValueError("queue_control.policy must be 'edf' when admission is enabled")

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
        stage_class_limits = _stage_class_limit_map(
            raw.get("stage_class_wip_limits"),
            field_name="queue_control.stage_class_wip_limits",
        )
        if num_stages is not None:
            invalid = sorted(stage_id for stage_id in stage_limits if stage_id >= num_stages)
            if invalid:
                raise ValueError(f"stage_wip_limits contains unavailable stage ids: {invalid}")
            invalid = sorted(stage_id for stage_id in stage_class_limits if stage_id >= num_stages)
            if invalid:
                raise ValueError(f"stage_class_wip_limits contains unavailable stage ids: {invalid}")

        admission = AdmissionControlConfig.from_mapping(raw.get("admission"))
        return cls(
            enabled=enabled,
            policy=policy,  # type: ignore[arg-type]
            global_wip_limit=_optional_limit(
                raw.get("global_wip_limit"),
                field_name="queue_control.global_wip_limit",
            ),
            stage_wip_limits=stage_limits,
            stage_class_wip_limits=stage_class_limits,
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
            admission=admission,
            online_allocator=OnlineAllocatorMetadata.from_mapping(raw.get("online_allocator")),
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


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """One calibrated ingress-admission evaluation."""

    admitted: bool
    request_id: str
    request_class: str
    phase: Literal["arrival", "recheck"]
    score: float | None
    gamma: float | None
    reason: str
    effective_k: int | None
    mu: float | None
    active_count: int
    queue_position: int
    remaining_budget_s: float | None


@dataclass(frozen=True, slots=True)
class RejectedStageDispatch:
    pending: PendingStageDispatch
    decision: AdmissionDecision


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
        self._active_stage_classes: dict[tuple[str, int], str] = {}
        # A dependency is monotone within one logical-request lifetime: an
        # upstream stage satisfies it while active and after successful
        # completion. Rollback never records completion, and cancellation
        # removes all completion history before a request id may be reused.
        self._completed_stages: set[tuple[str, int]] = set()
        self._observed_initial_requests: set[str] = set()
        self._arrivals_by_class_total: Counter[str] = Counter()
        self._config_generation = 0
        self._enqueued_total = 0
        self._dispatch_attempts_total = 0
        self._dispatch_failures_total = 0
        self._cancelled_total = 0
        self._queue_wait_s_total = 0.0
        self._queue_wait_s_max = 0.0
        self._admission_admitted_total = 0
        self._admission_rejected_total = 0
        self._admission_recheck_passed_total = 0
        self._admission_reason_counts: Counter[str] = Counter()
        self._recent_admission_decisions: deque[dict[str, Any]] = deque(maxlen=128)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def active(self) -> bool:
        """Whether either queue credits or ingress admission needs the path."""

        return self.enabled or self.config.admission.enabled

    def configure(self, config: QueueControlConfig) -> bool:
        if config == self.config:
            return False
        current_update = self.config.online_allocator
        incoming_update = config.online_allocator
        if current_update is not None and incoming_update is not None:
            if incoming_update.revision < current_update.revision:
                raise ValueError(
                    "queue_control.online_allocator.revision must not decrease "
                    f"({incoming_update.revision} < {current_update.revision})"
                )
            if incoming_update.revision == current_update.revision:
                raise ValueError(
                    "queue_control fields changed without advancing queue_control.online_allocator.revision"
                )
        self.config = config
        self._config_generation += 1
        return True

    def enqueue(self, pending: PendingStageDispatch) -> AdmissionDecision | None:
        if pending.stage_id < 0 or pending.stage_id >= self.num_stages:
            raise ValueError(f"stage_id {pending.stage_id} is outside [0, {self.num_stages})")
        self._stamp_pending(pending)
        decision = self._evaluate_arrival_admission(pending)
        if decision is not None:
            self._record_admission_decision(decision)
            if not decision.admitted:
                self._admission_rejected_total += 1
                return decision
            self._admission_admitted_total += 1
        self._pending.append(pending)
        return decision

    def _stamp_pending(self, pending: PendingStageDispatch) -> None:
        if (
            pending.starts_request
            and pending.stage_id == 0
            and pending.logical_request_id not in self._observed_initial_requests
        ):
            self._observed_initial_requests.add(pending.logical_request_id)
            self._arrivals_by_class_total[pending.metadata.request_class] += 1
        pending.sequence = self._next_sequence
        self._next_sequence += 1
        pending.enqueued_monotonic_s = self._clock()
        self._enqueued_total += 1

    def requires_queue(self, pending: PendingStageDispatch) -> bool:
        """Return whether this dispatch competes for an enforced credit.

        End-to-end request limits and admission only gate the first stage
        submission. Later stages stay on the immediate dispatch path unless a
        dependency, stage, or stage-class WIP limit requires queuing. This
        avoids serializing every pipeline edge behind a class-level controller.
        """
        if not self.active:
            return False
        if (
            self.config.admission.enabled
            and pending.starts_request
            and pending.stage_id == 0
            and pending.metadata.request_class in self.config.admission.classes
        ):
            return True
        if not self._dependency_satisfied(pending):
            return True
        if pending.logical_request_id not in self._active_requests and not pending.starts_request:
            return True
        if not self.enabled:
            return False
        stage_key = (pending.request_id, pending.stage_id)
        if stage_key not in self._active_stages:
            if pending.stage_id in self.config.stage_wip_limits:
                return True
            stage_class_limits = self.config.stage_class_wip_limits.get(pending.stage_id, {})
            if self._request_class(pending) in stage_class_limits:
                return True
        if pending.logical_request_id in self._active_requests:
            return False
        return (
            self.config.global_wip_limit is not None
            or pending.metadata.path in self.config.path_wip_limits
            or pending.metadata.request_class in self.config.class_wip_limits
        )

    def _request_counts(self) -> tuple[Counter[str], Counter[str]]:
        path_counts = Counter(metadata.path for metadata in self._active_requests.values())
        class_counts = Counter(metadata.request_class for metadata in self._active_requests.values())
        return path_counts, class_counts

    def _active_class_count(self, request_class: str) -> int:
        return sum(metadata.request_class == request_class for metadata in self._active_requests.values())

    def _request_class(self, pending: PendingStageDispatch) -> str:
        metadata = self._active_requests.get(pending.logical_request_id, pending.metadata)
        return metadata.request_class

    def _dependency_satisfied(self, pending: PendingStageDispatch) -> bool:
        required_stage_id = pending.required_active_stage_id
        if required_stage_id is None:
            return True
        dependency = (pending.request_id, required_stage_id)
        return dependency in self._active_stages or dependency in self._completed_stages

    @staticmethod
    def _admission_order_key(pending: PendingStageDispatch) -> tuple[float, int]:
        deadline = pending.metadata.deadline_monotonic_s
        return (math.inf if deadline is None else deadline, pending.sequence)

    def _admission_candidates(self, request_class: str) -> list[PendingStageDispatch]:
        return sorted(
            (
                pending
                for pending in self._pending
                if pending.starts_request
                and pending.stage_id == 0
                and pending.logical_request_id not in self._active_requests
                and pending.metadata.request_class == request_class
            ),
            key=self._admission_order_key,
        )

    def _evaluate_admission(
        self,
        pending: PendingStageDispatch,
        *,
        queue_position: int,
        phase: Literal["arrival", "recheck"],
    ) -> AdmissionDecision | None:
        admission = self.config.admission
        if not admission.enabled or not pending.starts_request or pending.stage_id != 0:
            return None
        class_config = admission.classes.get(pending.metadata.request_class)
        if class_config is None:
            return None

        active_count = self._active_class_count(pending.metadata.request_class)
        deadline = pending.metadata.deadline_monotonic_s
        remaining_budget_s = None if deadline is None else deadline - self._clock()
        if class_config.effective_k == 0:
            score = 0.0
            reason = "zero_effective_k"
            admitted = False
        elif active_count > class_config.effective_k:
            score = 0.0
            reason = "active_above_effective_k"
            admitted = False
        elif deadline is None:
            return AdmissionDecision(
                admitted=True,
                request_id=pending.logical_request_id,
                request_class=pending.metadata.request_class,
                phase=phase,
                score=None,
                gamma=class_config.gamma,
                reason="no_deadline",
                effective_k=class_config.effective_k,
                mu=class_config.mu,
                active_count=active_count,
                queue_position=queue_position,
                remaining_budget_s=None,
            )
        elif remaining_budget_s is not None and remaining_budget_s < 0:
            score = 0.0
            reason = "deadline_expired"
            admitted = False
        else:
            assert remaining_budget_s is not None
            score = erlang_empirical_admission_score(
                remaining_budget_s,
                effective_k=class_config.effective_k,
                mu=class_config.mu,
                active_count=active_count,
                queue_position=queue_position,
                service_samples_s=class_config.service_samples_s,
            )
            admitted = score >= class_config.gamma
            reason = "score_pass" if admitted else "score_below_gamma"
        return AdmissionDecision(
            admitted=admitted,
            request_id=pending.logical_request_id,
            request_class=pending.metadata.request_class,
            phase=phase,
            score=score,
            gamma=class_config.gamma,
            reason=reason,
            effective_k=class_config.effective_k,
            mu=class_config.mu,
            active_count=active_count,
            queue_position=queue_position,
            remaining_budget_s=remaining_budget_s,
        )

    def _evaluate_arrival_admission(self, pending: PendingStageDispatch) -> AdmissionDecision | None:
        candidates = [*self._admission_candidates(pending.metadata.request_class), pending]
        candidates.sort(key=self._admission_order_key)
        queue_position = 0
        for candidate in candidates:
            if candidate is pending:
                return self._evaluate_admission(
                    pending,
                    queue_position=queue_position,
                    phase="arrival",
                )
            prior_decision = self._evaluate_admission(
                candidate,
                queue_position=queue_position,
                phase="recheck",
            )
            if prior_decision is None or prior_decision.admitted:
                queue_position += 1
        raise AssertionError("arrival candidate disappeared from its tentative class queue")

    def _record_admission_decision(self, decision: AdmissionDecision) -> None:
        self._admission_reason_counts[decision.reason] += 1
        self._recent_admission_decisions.append(
            {
                "request_id": decision.request_id,
                "request_class": decision.request_class,
                "phase": decision.phase,
                "admitted": decision.admitted,
                "score": decision.score,
                "gamma": decision.gamma,
                "reason": decision.reason,
                "effective_k": decision.effective_k,
                "mu": decision.mu,
                "active_count": decision.active_count,
                "queue_position": decision.queue_position,
                "remaining_budget_s": decision.remaining_budget_s,
            }
        )

    def recheck_admission(self) -> list[RejectedStageDispatch]:
        """Re-evaluate waiting ingress requests after queue/config changes.

        Requests are swept in per-class absolute-deadline order. Rejected
        requests do not consume a position for later requests in the same
        sweep, matching the accepted-waiting queue in the model.
        """

        if not self.config.admission.enabled:
            return []
        rejected: list[RejectedStageDispatch] = []
        rejected_sequences: set[int] = set()
        classes = {
            pending.metadata.request_class
            for pending in self._pending
            if pending.starts_request and pending.stage_id == 0
        }
        for request_class in sorted(classes):
            queue_position = 0
            for pending in self._admission_candidates(request_class):
                decision = self._evaluate_admission(
                    pending,
                    queue_position=queue_position,
                    phase="recheck",
                )
                if decision is None:
                    queue_position += 1
                    continue
                self._record_admission_decision(decision)
                if decision.admitted:
                    self._admission_recheck_passed_total += 1
                    queue_position += 1
                    continue
                self._admission_rejected_total += 1
                rejected_sequences.add(pending.sequence)
                rejected.append(RejectedStageDispatch(pending=pending, decision=decision))
        if rejected_sequences:
            self._pending = [pending for pending in self._pending if pending.sequence not in rejected_sequences]
        return rejected

    def _blocked_reasons(self, pending: PendingStageDispatch) -> tuple[str, ...]:
        if not self.active:
            return ()
        reasons: list[str] = []
        if not self._dependency_satisfied(pending):
            reasons.append("dependency")
        if self.enabled:
            stage_key = (pending.request_id, pending.stage_id)
            if stage_key not in self._active_stages:
                stage_limit = self.config.stage_wip_limits.get(pending.stage_id)
                if stage_limit is not None:
                    active = sum(stage_id == pending.stage_id for _, stage_id in self._active_stages)
                    if active >= stage_limit:
                        reasons.append("stage")
                request_class = self._request_class(pending)
                stage_class_limit = self.config.stage_class_wip_limits.get(pending.stage_id, {}).get(request_class)
                if stage_class_limit is not None:
                    active = sum(
                        stage_id == pending.stage_id and active_class == request_class
                        for (_, stage_id), active_class in self._active_stage_classes.items()
                    )
                    if active >= stage_class_limit:
                        reasons.append("stage_class")

        if pending.logical_request_id not in self._active_requests:
            if not pending.starts_request:
                reasons.append("request")
                return tuple(reasons)
            path_counts, class_counts = self._request_counts()
            if self.enabled:
                global_limit = self.config.global_wip_limit
                if global_limit is not None and len(self._active_requests) >= global_limit:
                    reasons.append("global")
                path_limit = self.config.path_wip_limits.get(pending.metadata.path)
                if path_limit is not None and path_counts[pending.metadata.path] >= path_limit:
                    reasons.append("path")
                class_limit = self.config.class_wip_limits.get(pending.metadata.request_class)
                if class_limit is not None and class_counts[pending.metadata.request_class] >= class_limit:
                    reasons.append("class")
            admission_class = self.config.admission.classes.get(pending.metadata.request_class)
            if (
                self.config.admission.enabled
                and pending.starts_request
                and pending.stage_id == 0
                and admission_class is not None
                and class_counts[pending.metadata.request_class] >= admission_class.effective_k
            ):
                reasons.append("admission_class")
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
        return self._acquire(pending)

    def acquire_immediate(self, pending: PendingStageDispatch) -> AcquiredStageDispatch:
        """Record an unqueued dispatch while preserving lease telemetry."""
        if pending.stage_id < 0 or pending.stage_id >= self.num_stages:
            raise ValueError(f"stage_id {pending.stage_id} is outside [0, {self.num_stages})")
        self._stamp_pending(pending)
        return self._acquire(pending)

    def _acquire(self, pending: PendingStageDispatch) -> AcquiredStageDispatch:
        acquired_request = pending.logical_request_id not in self._active_requests
        acquired_stage = (pending.request_id, pending.stage_id) not in self._active_stages
        if acquired_request:
            self._active_requests[pending.logical_request_id] = pending.metadata
        self._request_to_logical[pending.request_id] = pending.logical_request_id
        if acquired_stage:
            stage_key = (pending.request_id, pending.stage_id)
            self._active_stages.add(stage_key)
            self._active_stage_classes[stage_key] = self._request_class(pending)

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
            stage_key = (pending.request_id, pending.stage_id)
            self._active_stages.discard(stage_key)
            self._active_stage_classes.pop(stage_key, None)
        if acquired.acquired_request:
            self._active_requests.pop(pending.logical_request_id, None)
        if not any(request_id == pending.request_id for request_id, _ in self._active_stages) and not any(
            request_id == pending.request_id for request_id, _ in self._completed_stages
        ):
            self._request_to_logical.pop(pending.request_id, None)
        self._dispatch_failures_total += 1

    def release_stage(self, request_id: str, stage_id: int) -> bool:
        key = (request_id, stage_id)
        if key not in self._active_stages:
            return False
        self._active_stages.remove(key)
        self._active_stage_classes.pop(key, None)
        self._completed_stages.add(key)
        return True

    def cancel_request(self, request_id: str) -> None:
        logical_id = self._request_to_logical.get(request_id, request_id)
        before = len(self._pending)
        if request_id == logical_id:
            self._pending = [item for item in self._pending if item.logical_request_id != logical_id]
            self._active_stages = {
                key for key in self._active_stages if self._request_to_logical.get(key[0], key[0]) != logical_id
            }
            self._active_stage_classes = {
                key: request_class
                for key, request_class in self._active_stage_classes.items()
                if self._request_to_logical.get(key[0], key[0]) != logical_id
            }
            self._completed_stages = {
                key for key in self._completed_stages if self._request_to_logical.get(key[0], key[0]) != logical_id
            }
            self._active_requests.pop(logical_id, None)
            for actual_id, mapped_logical in list(self._request_to_logical.items()):
                if mapped_logical == logical_id:
                    self._request_to_logical.pop(actual_id, None)
            self._observed_initial_requests.discard(logical_id)
        else:
            self._pending = [item for item in self._pending if item.request_id != request_id]
            self._active_stages = {key for key in self._active_stages if key[0] != request_id}
            self._active_stage_classes = {
                key: request_class for key, request_class in self._active_stage_classes.items() if key[0] != request_id
            }
            self._completed_stages = {key for key in self._completed_stages if key[0] != request_id}
            self._request_to_logical.pop(request_id, None)
        self._cancelled_total += before - len(self._pending)

    def snapshot(self) -> dict[str, Any]:
        path_counts, class_counts = self._request_counts()
        stage_counts = Counter(stage_id for _, stage_id in self._active_stages)
        stage_class_counts = Counter(
            (stage_id, request_class) for (_, stage_id), request_class in self._active_stage_classes.items()
        )
        queued_stage_counts = Counter(item.stage_id for item in self._pending)
        queued_stage_class_counts = Counter((item.stage_id, self._request_class(item)) for item in self._pending)
        queued_class_counts = Counter(
            item.metadata.request_class
            for item in self._pending
            if item.starts_request and item.stage_id == 0 and item.logical_request_id not in self._active_requests
        )
        blocked = Counter(reason for pending in self._pending for reason in self._blocked_reasons(pending))
        now = self._clock()
        return {
            "enabled": self.enabled,
            "policy": self.config.policy,
            "config_generation": self._config_generation,
            "global_wip_limit": self.config.global_wip_limit,
            "stage_wip_limits": {str(key): value for key, value in sorted(self.config.stage_wip_limits.items())},
            "stage_class_wip_limits": {
                str(stage_id): dict(sorted(class_limits.items()))
                for stage_id, class_limits in sorted(self.config.stage_class_wip_limits.items())
            },
            "path_wip_limits": dict(sorted(self.config.path_wip_limits.items())),
            "class_wip_limits": dict(sorted(self.config.class_wip_limits.items())),
            "online_allocator": (
                None if self.config.online_allocator is None else self.config.online_allocator.to_snapshot()
            ),
            "admission": {
                "enabled": self.config.admission.enabled,
                "score_method": self.config.admission.score_method,
                "classes": {
                    request_class: {
                        "effective_k": class_config.effective_k,
                        "mu": class_config.mu,
                        "gamma": class_config.gamma,
                        "service_sample_count": len(class_config.service_samples_s),
                    }
                    for request_class, class_config in sorted(self.config.admission.classes.items())
                },
                "admitted_total": self._admission_admitted_total,
                "rejected_total": self._admission_rejected_total,
                "recheck_passed_total": self._admission_recheck_passed_total,
                "decision_reason_counts": dict(sorted(self._admission_reason_counts.items())),
                "recent_decisions": list(self._recent_admission_decisions),
            },
            "active_requests": len(self._active_requests),
            "active_by_stage": {str(key): value for key, value in sorted(stage_counts.items())},
            "active_by_stage_class": {
                str(stage_id): {
                    request_class: stage_class_counts[stage_id, request_class]
                    for _, request_class in sorted(key for key in stage_class_counts if key[0] == stage_id)
                }
                for stage_id in sorted({stage_id for stage_id, _ in stage_class_counts})
            },
            "active_by_path": dict(sorted(path_counts.items())),
            "active_by_class": dict(sorted(class_counts.items())),
            "queued_requests": len(self._pending),
            "queued_by_stage": {str(key): value for key, value in sorted(queued_stage_counts.items())},
            "queued_by_stage_class": {
                str(stage_id): {
                    request_class: queued_stage_class_counts[stage_id, request_class]
                    for _, request_class in sorted(key for key in queued_stage_class_counts if key[0] == stage_id)
                }
                for stage_id in sorted({stage_id for stage_id, _ in queued_stage_class_counts})
            },
            "queued_by_class": dict(sorted(queued_class_counts.items())),
            "arrivals_by_class_total": dict(sorted(self._arrivals_by_class_total.items())),
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
