#!/usr/bin/env python3
"""CPU microbenchmark for the D1 admission hot path.

Run from the repository root:

    python tests/benchmarks/benchmark_queue_control_admission.py
"""

from __future__ import annotations

import json
import statistics
import time

from vllm_omni.engine.queue_control import (
    AdmissionClassConfig,
    AdmissionControlConfig,
    PendingStageDispatch,
    QueueControlConfig,
    RequestSchedulingMetadata,
    RuntimeQueueController,
    erlang_empirical_admission_score,
    erlang_empirical_admission_score_reference,
)


async def _dispatch() -> bool:
    return True


def _pending(request_id: str, deadline: float) -> PendingStageDispatch:
    return PendingStageDispatch(
        request_id=request_id,
        logical_request_id=request_id,
        stage_id=0,
        metadata=RequestSchedulingMetadata(
            request_class="interactive",
            path="text",
            deadline_monotonic_s=deadline,
        ),
        dispatch=_dispatch,
        operation="benchmark",
        starts_request=True,
    )


def _median_us(function, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def run_cell(queue_size: int, sample_count: int) -> dict[str, float | int]:
    service_samples = tuple(0.05 + 1.95 * index / max(sample_count - 1, 1) for index in range(sample_count))
    class_config = AdmissionClassConfig(
        effective_k=8,
        mu=2.0,
        service_samples_s=service_samples,
        gamma=0.1,
    )
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            policy="edf",
            admission=AdmissionControlConfig(
                enabled=True,
                classes={"interactive": class_config},
            ),
        ),
        clock=lambda: 0.0,
    )
    for index in range(queue_size):
        pending = _pending(f"waiting-{index}", 10_000.0 + index)
        controller._stamp_pending(pending)
        controller._pending.append(pending)
    newcomer = _pending("newcomer", 20_000.0)
    controller._stamp_pending(newcomer)

    def legacy_vectorized_arrival() -> None:
        queue_position = 0
        candidates = [*controller._admission_candidates("interactive"), newcomer]
        candidates.sort(key=controller._admission_order_key)
        for candidate in candidates:
            if candidate is newcomer:
                controller._evaluate_admission(
                    newcomer,
                    queue_position=queue_position,
                    phase="arrival",
                    now_monotonic_s=0.0,
                )
                return
            prior = controller._evaluate_admission(
                candidate,
                queue_position=queue_position,
                phase="recheck",
                now_monotonic_s=0.0,
            )
            if prior is None or prior.admitted:
                queue_position += 1
        raise AssertionError("newcomer missing from legacy sweep")

    remaining_budget_s = 2.0 + (queue_size + 1) / 16.0
    score_kwargs = {
        "effective_k": 8,
        "mu": 2.0,
        "active_count": 8,
        "queue_position": queue_size,
        "service_samples_s": service_samples,
    }
    erlang_empirical_admission_score(remaining_budget_s, **score_kwargs)
    controller._evaluate_arrival_admission(newcomer)
    legacy_vectorized_arrival()

    vector_us = _median_us(
        lambda: erlang_empirical_admission_score(remaining_budget_s, **score_kwargs),
        repeats=20,
    )
    scalar_us = _median_us(
        lambda: erlang_empirical_admission_score_reference(remaining_budget_s, **score_kwargs),
        repeats=3,
    )
    arrival_us = _median_us(
        lambda: controller._evaluate_arrival_admission(newcomer),
        repeats=20,
    )
    legacy_arrival_us = _median_us(legacy_vectorized_arrival, repeats=3)
    return {
        "queue_size": queue_size,
        "sample_count": sample_count,
        "vector_score_us": vector_us,
        "scalar_score_us": scalar_us,
        "score_speedup": scalar_us / vector_us,
        "successful_arrival_us": arrival_us,
        "legacy_vectorized_arrival_us": legacy_arrival_us,
        "arrival_speedup": legacy_arrival_us / arrival_us,
    }


def main() -> None:
    results = [
        run_cell(queue_size, sample_count) for queue_size in (10, 50, 100, 500) for sample_count in (64, 576, 2000)
    ]
    print(json.dumps({"schema_version": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()
