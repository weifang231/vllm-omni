# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

import pytest

from vllm_omni.engine.queue_control import (
    AdmissionClassConfig,
    AdmissionControlConfig,
    OnlineAllocatorMetadata,
    PendingStageDispatch,
    QueueControlConfig,
    RequestSchedulingMetadata,
    RuntimeQueueController,
    erlang_empirical_admission_score,
    erlang_wait_cdf,
    scheduling_kwargs_from_headers,
)


async def _dispatch() -> bool:
    return True


def _pending(
    request_id: str,
    *,
    logical_request_id: str | None = None,
    deadline: float | None = None,
    stage_id: int = 0,
    starts_request: bool = True,
    request_class: str = "default",
    path: str = "default",
    required_active_stage_id: int | None = None,
    dispatch: Callable[[], Awaitable[bool]] = _dispatch,
) -> PendingStageDispatch:
    return PendingStageDispatch(
        request_id=request_id,
        logical_request_id=logical_request_id or request_id,
        stage_id=stage_id,
        metadata=RequestSchedulingMetadata(
            request_class=request_class,
            path=path,
            deadline_monotonic_s=deadline,
        ),
        dispatch=dispatch,
        operation="test",
        starts_request=starts_request,
        required_active_stage_id=required_active_stage_id,
    )


def _admission_config(
    *,
    effective_k: int = 1,
    mu: float = 2.0,
    service_samples_s: tuple[float, ...] = (0.0,),
    gamma: float = 0.75,
) -> QueueControlConfig:
    return QueueControlConfig(
        policy="edf",
        admission=AdmissionControlConfig(
            enabled=True,
            classes={
                "interactive": AdmissionClassConfig(
                    effective_k=effective_k,
                    mu=mu,
                    service_samples_s=service_samples_s,
                    gamma=gamma,
                )
            },
        ),
    )


def test_queue_control_config_defaults_and_validation() -> None:
    assert QueueControlConfig.from_document({}) == QueueControlConfig()

    config = QueueControlConfig.from_document(
        {
            "queue_control": {
                "policy": "EDF",
                "global_wip_limit": 3,
                "stage_wip_limits": {"0": 2, "1": 1},
                "stage_class_wip_limits": {
                    "0": {"interactive": 2},
                    "1": {"batch": 1, "interactive": 1},
                },
                "path_wip_limits": {"audio": 2},
                "class_wip_limits": {"interactive": 1},
            }
        },
        num_stages=2,
    )
    assert config.enabled
    assert config.policy == "edf"
    assert config.stage_wip_limits == {0: 2, 1: 1}
    assert config.stage_class_wip_limits == {
        0: {"interactive": 2},
        1: {"batch": 1, "interactive": 1},
    }

    with pytest.raises(ValueError, match="must not be null"):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_wip_limits": {"0": None}}},
            num_stages=1,
        )
    with pytest.raises(ValueError, match="unavailable stage ids"):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_wip_limits": {"1": 1}}},
            num_stages=1,
        )
    with pytest.raises(ValueError, match=r"stage_class_wip_limits\['0'\] must be a JSON object"):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_class_wip_limits": {"0": 1}}},
            num_stages=1,
        )
    with pytest.raises(ValueError, match="must not be null"):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_class_wip_limits": {"0": {"interactive": None}}}},
            num_stages=1,
        )
    with pytest.raises(ValueError, match="stage_class_wip_limits contains unavailable stage ids"):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_class_wip_limits": {"1": {"interactive": 1}}}},
            num_stages=1,
        )


def test_online_allocator_metadata_is_parsed_and_acknowledged_in_snapshot() -> None:
    metadata = {
        "schema_version": 1,
        "revision": 7,
        "source_runtime_id": "runtime-a",
        "source_snapshot_sequence": 13,
        "source_config_generation": 2,
        "profile_fingerprint": "a" * 64,
    }
    config = QueueControlConfig.from_document(
        {
            "queue_control": {
                "enabled": True,
                "class_wip_limits": {"audio": 3, "text": 1},
                "online_allocator": metadata,
            }
        }
    )
    assert config.online_allocator == OnlineAllocatorMetadata(
        revision=7,
        source_runtime_id="runtime-a",
        source_snapshot_sequence=13,
        source_config_generation=2,
        profile_fingerprint="a" * 64,
    )

    controller = RuntimeQueueController(num_stages=1, config=config)
    assert controller.snapshot()["online_allocator"] == metadata


def test_online_allocator_revision_prevents_stale_or_torn_logical_updates() -> None:
    def config(revision: int, audio_limit: int) -> QueueControlConfig:
        return QueueControlConfig(
            enabled=True,
            class_wip_limits={"audio": audio_limit},
            online_allocator=OnlineAllocatorMetadata(
                revision=revision,
                source_runtime_id="runtime-a",
                source_snapshot_sequence=revision,
                source_config_generation=revision - 1,
                profile_fingerprint="b" * 64,
            ),
        )

    controller = RuntimeQueueController(num_stages=1, config=config(2, 2))
    assert controller.configure(config(3, 3))
    assert controller.snapshot()["class_wip_limits"] == {"audio": 3}
    assert controller.snapshot()["config_generation"] == 1

    with pytest.raises(ValueError, match="must not decrease"):
        controller.configure(config(2, 1))
    with pytest.raises(ValueError, match="without advancing"):
        controller.configure(config(3, 1))
    assert controller.snapshot()["class_wip_limits"] == {"audio": 3}
    assert controller.snapshot()["config_generation"] == 1


def test_admission_config_parses_calibrated_class_inputs() -> None:
    config = QueueControlConfig.from_document(
        {
            "queue_control": {
                "enabled": False,
                "policy": "edf",
                "admission": {
                    "enabled": True,
                    "score_method": "erlang_empirical",
                    "classes": {
                        "interactive": {
                            "effective_k": 4,
                            "mu": 0.25,
                            "service_samples_s": [0.4, 0.8],
                            "gamma": 0.9,
                        }
                    },
                },
            }
        }
    )
    assert not config.enabled
    assert config.admission.enabled
    assert config.admission.classes["interactive"] == AdmissionClassConfig(
        effective_k=4,
        mu=0.25,
        service_samples_s=(0.4, 0.8),
        gamma=0.9,
    )

    with pytest.raises(ValueError, match="service_samples_s"):
        QueueControlConfig.from_document(
            {
                "queue_control": {
                    "admission": {
                        "classes": {
                            "interactive": {
                                "effective_k": 1,
                                "mu": 1.0,
                                "service_samples_s": [],
                                "gamma": 0.9,
                            }
                        }
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="policy must be 'edf'"):
        QueueControlConfig.from_document(
            {
                "queue_control": {
                    "admission": {
                        "classes": {
                            "interactive": {
                                "effective_k": 1,
                                "mu": 1.0,
                                "service_samples_s": [0.5],
                                "gamma": 0.9,
                            }
                        }
                    }
                }
            }
        )


def test_erlang_empirical_formula_matches_closed_form() -> None:
    assert erlang_wait_cdf(
        1.0,
        effective_k=2,
        mu=1.0,
        required_returns=1,
    ) == pytest.approx(1.0 - math.exp(-2.0))
    assert erlang_wait_cdf(
        1.0,
        effective_k=2,
        mu=1.0,
        required_returns=2,
    ) == pytest.approx(1.0 - math.exp(-2.0) * 3.0)

    score = erlang_empirical_admission_score(
        2.0,
        effective_k=2,
        mu=1.0,
        active_count=2,
        queue_position=0,
        service_samples_s=(0.5, 1.0),
    )
    expected = ((1.0 - math.exp(-3.0)) + (1.0 - math.exp(-2.0))) / 2.0
    assert score == pytest.approx(expected)


def test_request_metadata_builds_absolute_deadline() -> None:
    metadata = RequestSchedulingMetadata.create(
        path="audio",
        first_output_deadline_s=0.4,
        now_monotonic_s=10.0,
    )
    assert metadata.request_class == "audio"
    assert metadata.path == "audio"
    assert metadata.deadline_monotonic_s == pytest.approx(10.4)

    with pytest.raises(ValueError, match="non-negative"):
        RequestSchedulingMetadata.create(first_output_deadline_s=-0.1)


def test_http_headers_require_explicit_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = {
        "X-VLLM-OMNI-REQUEST-CLASS": "interactive",
        "x-vllm-omni-request-path": "audio",
        "x-vllm-omni-first-output-deadline-ms": "400",
    }
    assert scheduling_kwargs_from_headers(headers) == {}

    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    assert scheduling_kwargs_from_headers(headers) == {
        "request_class": "interactive",
        "request_path": "audio",
        "first_output_deadline_s": 0.4,
    }

    with pytest.raises(ValueError, match="finite and non-negative"):
        scheduling_kwargs_from_headers(
            {"x-vllm-omni-first-output-deadline-ms": "-1"},
            trusted=True,
        )


def test_edf_reorders_only_ready_requests_and_is_stable() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(enabled=True, policy="edf", global_wip_limit=1),
    )
    controller.enqueue(_pending("running", deadline=20.0))
    assert controller.pop_ready().pending.request_id == "running"  # type: ignore[union-attr]

    controller.enqueue(_pending("late", deadline=30.0))
    controller.enqueue(_pending("early-a", deadline=10.0))
    controller.enqueue(_pending("early-b", deadline=10.0))
    assert controller.pop_ready() is None

    controller.cancel_request("running")
    assert controller.pop_ready().pending.request_id == "early-a"  # type: ignore[union-attr]
    controller.cancel_request("early-a")
    assert controller.pop_ready().pending.request_id == "early-b"  # type: ignore[union-attr]


def test_stage_credit_allows_updates_but_blocks_new_wip() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(enabled=True, stage_wip_limits={0: 1}),
    )
    controller.enqueue(_pending("r1"))
    assert controller.pop_ready() is not None

    controller.enqueue(_pending("r1", starts_request=False))
    assert controller.pop_ready() is not None

    controller.enqueue(_pending("r2"))
    assert controller.pop_ready() is None
    assert controller.release_stage("r1", 0)
    assert controller.pop_ready().pending.request_id == "r2"  # type: ignore[union-attr]


def test_stage_class_credit_blocks_only_matching_stage_and_class() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 1}},
        ),
    )
    for request_id, request_class in (
        ("speech-1", "speech"),
        ("speech-2", "speech"),
        ("text-1", "text"),
    ):
        controller.acquire_immediate(_pending(request_id, request_class=request_class))

    controller.enqueue(_pending("speech-1", stage_id=1, starts_request=False, request_class="speech"))
    assert controller.pop_ready() is not None
    controller.enqueue(_pending("speech-2", stage_id=1, starts_request=False, request_class="speech"))
    assert controller.pop_ready() is None

    text = _pending("text-1", stage_id=1, starts_request=False, request_class="text")
    assert not controller.requires_queue(text)
    controller.acquire_immediate(text)

    snapshot = controller.snapshot()
    assert snapshot["active_by_stage_class"] == {
        "0": {"speech": 2, "text": 1},
        "1": {"speech": 1, "text": 1},
    }
    assert snapshot["queued_by_stage_class"] == {"1": {"speech": 1}}
    assert snapshot["blocked_by_limit"] == {"stage_class": 1}

    assert controller.release_stage("speech-1", 1)
    assert controller.pop_ready().pending.request_id == "speech-2"  # type: ignore[union-attr]


def test_stage_class_credit_is_inert_when_queue_control_is_disabled() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=False,
            stage_class_wip_limits={0: {"speech": 0}},
        ),
    )
    pending = _pending("stock", request_class="speech")
    assert not controller.requires_queue(pending)
    controller.acquire_immediate(pending)
    assert controller.snapshot()["active_by_stage_class"] == {"0": {"speech": 1}}


def test_stage_class_live_reconfiguration_and_reduction_are_nonpreemptive() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 1}},
        ),
    )
    for request_id in ("r1", "r2", "r3"):
        controller.acquire_immediate(_pending(request_id, request_class="speech"))

    controller.enqueue(_pending("r1", stage_id=1, starts_request=False, request_class="speech"))
    assert controller.pop_ready() is not None
    controller.enqueue(_pending("r2", stage_id=1, starts_request=False, request_class="speech"))
    assert controller.pop_ready() is None

    assert controller.configure(
        QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 2}},
        )
    )
    assert controller.pop_ready().pending.request_id == "r2"  # type: ignore[union-attr]

    assert controller.configure(
        QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 1}},
        )
    )
    controller.enqueue(_pending("r3", stage_id=1, starts_request=False, request_class="speech"))
    assert controller.pop_ready() is None
    assert controller.snapshot()["active_by_stage_class"]["1"] == {"speech": 2}

    assert controller.release_stage("r1", 1)
    assert controller.pop_ready() is None
    assert controller.release_stage("r2", 1)
    assert controller.pop_ready().pending.request_id == "r3"  # type: ignore[union-attr]


def test_stage_class_accounting_uses_logical_request_class_and_parent_cleanup() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 1}},
        ),
    )
    controller.acquire_immediate(_pending("parent", request_class="speech"))

    controller.enqueue(
        _pending(
            "companion-1",
            logical_request_id="parent",
            stage_id=1,
            starts_request=False,
            request_class="untrusted-override",
        )
    )
    assert controller.pop_ready() is not None
    controller.enqueue(
        _pending(
            "companion-2",
            logical_request_id="parent",
            stage_id=1,
            starts_request=False,
            request_class="untrusted-override",
        )
    )
    assert controller.pop_ready() is None

    snapshot = controller.snapshot()
    assert snapshot["active_by_stage_class"]["1"] == {"speech": 1}
    assert snapshot["queued_by_stage_class"]["1"] == {"speech": 1}

    controller.cancel_request("parent")
    snapshot = controller.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["active_by_stage"] == {}
    assert snapshot["active_by_stage_class"] == {}
    assert snapshot["queued_requests"] == 0
    assert snapshot["queued_by_stage_class"] == {}


def test_path_and_class_credits_are_end_to_end_and_nonpreemptive() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(
            enabled=True,
            path_wip_limits={"audio": 2},
            class_wip_limits={"interactive": 2},
        ),
    )
    for request_id in ("r1", "r2"):
        controller.enqueue(
            _pending(
                request_id,
                path="audio",
                request_class="interactive",
            )
        )
        assert controller.pop_ready() is not None

    controller.configure(
        QueueControlConfig(
            enabled=True,
            path_wip_limits={"audio": 1},
            class_wip_limits={"interactive": 1},
        )
    )
    controller.enqueue(
        _pending(
            "r3",
            path="audio",
            request_class="interactive",
        )
    )
    assert controller.pop_ready() is None
    assert controller.snapshot()["active_requests"] == 2

    controller.cancel_request("r1")
    assert controller.pop_ready() is None
    controller.cancel_request("r2")
    assert controller.pop_ready().pending.request_id == "r3"  # type: ignore[union-attr]


def test_dependency_and_cancellation_release_all_request_leases() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(enabled=True),
    )
    controller.enqueue(
        _pending(
            "r1",
            stage_id=1,
            starts_request=False,
            required_active_stage_id=0,
        )
    )
    assert controller.pop_ready() is None

    controller.enqueue(_pending("r1", stage_id=0))
    assert controller.pop_ready().pending.stage_id == 0  # type: ignore[union-attr]
    assert controller.pop_ready().pending.stage_id == 1  # type: ignore[union-attr]
    controller.cancel_request("r1")

    snapshot = controller.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["active_by_stage"] == {}
    assert snapshot["queued_requests"] == 0


def test_request_limits_only_queue_initial_dispatches() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            policy="edf",
            global_wip_limit=8,
            class_wip_limits={"audio": 7},
        ),
    )
    initial = _pending("r1", stage_id=0, request_class="audio")
    assert controller.requires_queue(initial)
    controller.enqueue(initial)
    assert controller.pop_ready() is not None

    downstream = _pending(
        "r1",
        stage_id=1,
        starts_request=False,
        request_class="audio",
    )
    assert not controller.requires_queue(downstream)
    acquired = controller.acquire_immediate(downstream)
    assert acquired.pending is downstream
    assert controller.snapshot()["active_by_stage"] == {"0": 1, "1": 1}

    controller.configure(
        QueueControlConfig(
            enabled=True,
            global_wip_limit=8,
            stage_wip_limits={2: 1},
        )
    )
    limited_stage = _pending("r1", stage_id=2, starts_request=False)
    assert controller.requires_queue(limited_stage)


def test_class_arrivals_count_each_logical_initial_request_once() -> None:
    controller = RuntimeQueueController(num_stages=3)
    initial = _pending("r1", request_class="audio")
    controller.acquire_immediate(initial)
    controller.acquire_immediate(
        _pending(
            "r1",
            stage_id=1,
            starts_request=False,
            request_class="audio",
        )
    )
    controller.acquire_immediate(
        _pending(
            "r1-update",
            stage_id=0,
            starts_request=False,
            request_class="audio",
        )
    )
    # A duplicate initial submission in the same request lifetime must not
    # turn one logical request into two offered arrivals.
    controller.acquire_immediate(_pending("r1", request_class="audio"))

    snapshot = controller.snapshot()
    assert snapshot["arrivals_by_class_total"] == {"audio": 1}
    assert snapshot["enqueued_total"] == 4

    controller.cancel_request("r1")
    controller.acquire_immediate(_pending("r1", request_class="audio"))
    assert controller.snapshot()["arrivals_by_class_total"] == {"audio": 2}


def test_queued_by_class_counts_only_waiting_initial_requests() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            global_wip_limit=1,
        ),
    )
    controller.enqueue(_pending("running", request_class="text"))
    assert controller.pop_ready() is not None
    controller.enqueue(_pending("waiting", request_class="audio"))
    controller.enqueue(
        _pending(
            "downstream",
            stage_id=1,
            starts_request=False,
            request_class="audio",
            required_active_stage_id=0,
        )
    )

    snapshot = controller.snapshot()
    assert snapshot["queued_requests"] == 2
    assert snapshot["queued_by_stage"] == {"0": 1, "1": 1}
    assert snapshot["queued_by_class"] == {"audio": 1}
    assert snapshot["arrivals_by_class_total"] == {"audio": 1, "text": 1}


def test_rejected_initial_request_is_an_offered_arrival_but_not_queued() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(effective_k=0, mu=0.0),
        clock=lambda: 10.0,
    )
    decision = controller.enqueue(_pending("rejected", request_class="interactive", deadline=20.0))
    assert decision is not None and not decision.admitted

    snapshot = controller.snapshot()
    assert snapshot["arrivals_by_class_total"] == {"interactive": 1}
    assert snapshot["queued_by_class"] == {}
    assert snapshot["queued_requests"] == 0


def test_admission_without_deadline_is_admitted_without_a_score() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(),
        clock=lambda: 10.0,
    )
    pending = _pending("no-deadline", request_class="interactive")
    assert controller.requires_queue(pending)
    decision = controller.enqueue(pending)
    assert decision is not None
    assert decision.admitted
    assert decision.score is None
    assert decision.reason == "no_deadline"
    assert controller.pop_ready() is not None


def test_admission_rejects_expired_deadline_and_zero_effective_k() -> None:
    expired = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(),
        clock=lambda: 10.0,
    )
    expired_decision = expired.enqueue(_pending("expired", request_class="interactive", deadline=9.0))
    assert expired_decision is not None
    assert not expired_decision.admitted
    assert expired_decision.score == 0.0
    assert expired_decision.reason == "deadline_expired"
    assert expired.snapshot()["queued_requests"] == 0

    stopped = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(effective_k=0, mu=0.0),
        clock=lambda: 10.0,
    )
    stopped_decision = stopped.enqueue(_pending("stopped", request_class="interactive", deadline=20.0))
    assert stopped_decision is not None
    assert not stopped_decision.admitted
    assert stopped_decision.reason == "zero_effective_k"


def test_admission_rechecks_deadline_position_after_queue_change() -> None:
    now = [0.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(),
        clock=lambda: now[0],
    )
    assert controller.enqueue(_pending("running", request_class="interactive", deadline=100.0)).admitted  # type: ignore[union-attr]
    assert controller.pop_ready() is not None

    late = controller.enqueue(_pending("late", request_class="interactive", deadline=1.0))
    assert late is not None and late.admitted
    early = controller.enqueue(_pending("early", request_class="interactive", deadline=0.8))
    assert early is not None and early.admitted

    rejected = controller.recheck_admission()
    assert [item.pending.request_id for item in rejected] == ["late"]
    assert rejected[0].decision.phase == "recheck"
    assert rejected[0].decision.queue_position == 1
    assert rejected[0].decision.reason == "score_below_gamma"
    assert controller.snapshot()["queued_requests"] == 1

    controller.cancel_request("running")
    assert controller.pop_ready().pending.request_id == "early"  # type: ignore[union-attr]
    controller.cancel_request("early")
    snapshot = controller.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["admission"]["rejected_total"] == 1
    assert snapshot["admission"]["decision_reason_counts"]["score_below_gamma"] == 1


def test_admission_only_mode_does_not_dispatch_update_before_initial() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(gamma=0.1),
        clock=lambda: 0.0,
    )
    controller.enqueue(_pending("running", request_class="interactive", deadline=100.0))
    assert controller.pop_ready() is not None

    controller.enqueue(_pending("waiting", request_class="interactive", deadline=100.0))
    controller.enqueue(
        _pending(
            "waiting",
            request_class="interactive",
            deadline=100.0,
            starts_request=False,
        )
    )
    assert controller.pop_ready() is None

    controller.cancel_request("running")
    assert controller.pop_ready().pending.starts_request is True  # type: ignore[union-attr]
    assert controller.pop_ready().pending.starts_request is False  # type: ignore[union-attr]


def test_arrival_position_excludes_waiters_that_now_fail_recheck() -> None:
    now = [0.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(mu=0.1, gamma=0.8),
        clock=lambda: now[0],
    )
    stale = controller.enqueue(_pending("stale", request_class="interactive", deadline=1.0))
    assert stale is not None and stale.admitted

    now[0] = 2.0
    newcomer = controller.enqueue(_pending("new", request_class="interactive", deadline=3.0))
    assert newcomer is not None and newcomer.admitted
    assert newcomer.queue_position == 0

    rejected = controller.recheck_admission()
    assert [item.pending.request_id for item in rejected] == ["stale"]
    assert controller.pop_ready().pending.request_id == "new"  # type: ignore[union-attr]


def test_admission_rechecks_after_effective_limit_change() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(effective_k=2, mu=2.0, gamma=0.5),
        clock=lambda: 0.0,
    )
    decision = controller.enqueue(_pending("waiting", request_class="interactive", deadline=10.0))
    assert decision is not None and decision.admitted

    controller.configure(_admission_config(effective_k=0, mu=0.0, gamma=0.5))
    rejected = controller.recheck_admission()
    assert len(rejected) == 1
    assert rejected[0].decision.reason == "zero_effective_k"


def test_admission_default_off_preserves_expired_request() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(),
        clock=lambda: 10.0,
    )
    decision = controller.enqueue(_pending("stock", request_class="interactive", deadline=1.0))
    assert decision is None
    assert controller.recheck_admission() == []
    assert controller.pop_ready().pending.request_id == "stock"  # type: ignore[union-attr]
