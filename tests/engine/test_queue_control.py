# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from vllm_omni.engine.queue_control import (
    PendingStageDispatch,
    QueueControlConfig,
    RequestSchedulingMetadata,
    RuntimeQueueController,
    scheduling_kwargs_from_headers,
)


async def _dispatch() -> bool:
    return True


def _pending(
    request_id: str,
    *,
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
        logical_request_id=request_id,
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


def test_queue_control_config_defaults_and_validation() -> None:
    assert QueueControlConfig.from_document({}) == QueueControlConfig()

    config = QueueControlConfig.from_document(
        {
            "queue_control": {
                "policy": "EDF",
                "global_wip_limit": 3,
                "stage_wip_limits": {"0": 2, "1": 1},
                "path_wip_limits": {"audio": 2},
                "class_wip_limits": {"interactive": 1},
            }
        },
        num_stages=2,
    )
    assert config.enabled
    assert config.policy == "edf"
    assert config.stage_wip_limits == {0: 2, 1: 1}

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
    assert scheduling_kwargs_from_headers(
        headers
    ) == {
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
