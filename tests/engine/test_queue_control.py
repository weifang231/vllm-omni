# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import math
import random
from collections.abc import Awaitable, Callable

import pytest

import vllm_omni.engine.queue_control as queue_control_module
from vllm_omni.engine.queue_control import (
    ADMISSION_DECISION_HISTORY_LIMIT,
    DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS,
    RECENT_STAGE_CANCELLATION_HISTORY_LIMIT,
    RECENT_STAGE_COMPLETION_HISTORY_LIMIT,
    AdmissionClassConfig,
    AdmissionControlConfig,
    OnlineAllocatorMetadata,
    PendingStageDispatch,
    QueueControlConfig,
    RequestSchedulingMetadata,
    RuntimeQueueController,
    StageBackpressureConfig,
    admission_threshold_profile_fingerprint,
    admission_threshold_table_digest,
    compile_erlang_empirical_admission_threshold_table,
    compile_erlang_empirical_admission_thresholds,
    erlang_empirical_admission_score,
    erlang_empirical_admission_score_reference,
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
    admission_correlation_id: str | None = None,
    required_active_stage_id: int | None = None,
    preserve_stage0_mm_cache_order: bool = False,
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
            admission_correlation_id=admission_correlation_id,
        ),
        dispatch=dispatch,
        operation="test",
        starts_request=starts_request,
        required_active_stage_id=required_active_stage_id,
        preserve_stage0_mm_cache_order=preserve_stage0_mm_cache_order,
    )


def _stage_backpressure(
    *,
    enabled: bool = True,
    upstream_stage_id: int = 0,
    downstream_stage_ids: tuple[int, ...] = (1, 2),
    request_class: str = "speech",
) -> StageBackpressureConfig:
    return StageBackpressureConfig(
        enabled=enabled,
        upstream_stage_id=upstream_stage_id,
        downstream_stage_ids=downstream_stage_ids,
        request_class=request_class,
    )


def _stage_backpressure_mapping(**overrides: object) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schema_version": 1,
        "enabled": True,
        "upstream_stage_id": 0,
        "downstream_stage_ids": [1, 2],
        "request_class": "speech",
    }
    mapping.update(overrides)
    return mapping


def _soft_stage0_config(
    *,
    stage_limit: int = 2,
    text_reservation: int = 1,
    speech_reservation: int = 1,
    policy: str = "fifo",
) -> QueueControlConfig:
    return QueueControlConfig(
        enabled=True,
        policy=policy,  # type: ignore[arg-type]
        stage_wip_limits={0: stage_limit},
        stage_class_wip_limits={
            0: {
                "text": text_reservation,
                "speech": speech_reservation,
            }
        },
        stage_class_wip_modes={0: "soft_reservation"},
    )


def _admission_config(
    *,
    effective_k: int = 1,
    mu: float = 2.0,
    service_samples_s: tuple[float, ...] = (0.0,),
    gamma: float = 0.75,
    enforce: bool = True,
) -> QueueControlConfig:
    return QueueControlConfig(
        policy="edf",
        admission=AdmissionControlConfig(
            enabled=True,
            enforce=enforce,
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


def _threshold_admission_config(
    *,
    effective_k: int = 1,
    mu: float = 2.0,
    service_samples_s: tuple[float, ...] = (0.0,),
    gamma: float = 0.75,
    max_required_returns: int = 8,
    enforce: bool = True,
) -> QueueControlConfig:
    table = compile_erlang_empirical_admission_threshold_table(
        "interactive",
        effective_k=effective_k,
        mu=mu,
        service_samples_s=service_samples_s,
        gamma=gamma,
        max_required_returns=max_required_returns,
    )
    return QueueControlConfig(
        policy="edf",
        admission=AdmissionControlConfig(
            enabled=True,
            enforce=enforce,
            score_method="erlang_empirical_threshold",
            classes={
                "interactive": AdmissionClassConfig(
                    effective_k=effective_k,
                    mu=mu,
                    service_samples_s=service_samples_s,
                    gamma=gamma,
                    max_required_returns=max_required_returns,
                    compiled_thresholds_s=table.thresholds_s,
                    threshold_profile_fingerprint=table.profile_fingerprint,
                    threshold_table_digest=table.table_digest,
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


def test_soft_stage_class_reservation_config_is_validated_and_fingerprinted() -> None:
    document = {
        "queue_control": {
            "enabled": True,
            "stage_wip_limits": {"0": 8},
            "stage_class_wip_limits": {"0": {"text": 6, "speech": 2}},
            "stage_class_wip_modes": {"0": "soft_reservation"},
        }
    }
    config = QueueControlConfig.from_document(document, num_stages=3)
    assert config.stage_class_wip_modes == {0: "soft_reservation"}
    assert config.semantic_mapping()["stage_class_wip_modes"] == {"0": "soft_reservation"}

    hard = QueueControlConfig(
        enabled=True,
        stage_wip_limits={0: 8},
        stage_class_wip_limits={0: {"text": 6, "speech": 2}},
    )
    assert config.fingerprint() != hard.fingerprint()

    with pytest.raises(ValueError, match="requires queue_control.stage_wip_limits"):
        QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"text": 1}},
            stage_class_wip_modes={0: "soft_reservation"},
        )
    with pytest.raises(ValueError, match="shares must not exceed"):
        QueueControlConfig(
            enabled=True,
            stage_wip_limits={0: 1},
            stage_class_wip_limits={0: {"text": 1, "speech": 1}},
            stage_class_wip_modes={0: "soft_reservation"},
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        QueueControlConfig(
            enabled=True,
            stage_wip_limits={0: 2},
            stage_class_wip_limits={0: {"text": 1, "speech": 1}},
            stage_class_wip_modes={0: "soft_reservation"},
            stage_backpressure=_stage_backpressure(),
        )


@pytest.mark.parametrize(
    ("modes", "error"),
    [
        ([], "must be a JSON object"),
        ({"0": 1}, "must be a string"),
        ({"0": "soft"}, "must be 'hard_limit' or 'soft_reservation'"),
        ({"1": "soft_reservation"}, "contains unavailable stage ids"),
    ],
)
def test_soft_stage_class_reservation_config_rejects_invalid_modes(
    modes: object,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        QueueControlConfig.from_document(
            {"queue_control": {"stage_class_wip_modes": modes}},
            num_stages=1,
        )


def test_stage_backpressure_config_is_normalized_and_snapshot_ready() -> None:
    config = QueueControlConfig.from_document(
        {
            "queue_control": {
                "enabled": True,
                "stage_class_wip_limits": {"0": {"speech": 2}},
                "stage_backpressure": _stage_backpressure_mapping(downstream_stage_ids=[2, 1]),
            }
        },
        num_stages=3,
    )

    assert config.stage_backpressure == _stage_backpressure()
    assert config.semantic_mapping()["stage_backpressure"] == _stage_backpressure_mapping()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"schema_version": True}, "schema_version must be 1"),
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"enabled": 1}, "enabled must be boolean"),
        ({"upstream_stage_id": True}, "upstream_stage_id must be a non-negative integer"),
        ({"upstream_stage_id": "0"}, "upstream_stage_id must be a non-negative integer"),
        ({"upstream_stage_id": 0.0}, "upstream_stage_id must be a non-negative integer"),
        ({"upstream_stage_id": -1}, "upstream_stage_id must be a non-negative integer"),
        ({"upstream_stage_id": 1, "downstream_stage_ids": [2]}, "upstream_stage_id must be 0"),
        ({"downstream_stage_ids": (1, 2)}, "downstream_stage_ids must be a JSON array"),
        ({"downstream_stage_ids": []}, "downstream_stage_ids must not be empty"),
        ({"downstream_stage_ids": [1, 1]}, "downstream_stage_ids must not contain duplicates"),
        ({"downstream_stage_ids": [0, 1]}, "must contain only stage ids greater than upstream_stage_id"),
        ({"downstream_stage_ids": [1, True]}, r"downstream_stage_ids\[1\] must be a non-negative integer"),
        ({"downstream_stage_ids": [1, "2"]}, r"downstream_stage_ids\[1\] must be a non-negative integer"),
        ({"downstream_stage_ids": [1, -1]}, r"downstream_stage_ids\[1\] must be a non-negative integer"),
        ({"request_class": 1}, "request_class must be a string"),
        ({"request_class": ""}, "request_class must be non-empty"),
    ],
)
def test_stage_backpressure_config_rejects_invalid_types_and_topology(
    overrides: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        QueueControlConfig.from_document(
            {
                "queue_control": {
                    "stage_backpressure": _stage_backpressure_mapping(**overrides),
                }
            },
            num_stages=3,
        )


def test_stage_backpressure_config_rejects_missing_unknown_and_unavailable_fields() -> None:
    missing = _stage_backpressure_mapping()
    missing.pop("request_class")
    with pytest.raises(ValueError, match="missing required fields"):
        QueueControlConfig.from_document({"queue_control": {"stage_backpressure": missing}}, num_stages=3)

    unknown = _stage_backpressure_mapping(extra=True)
    with pytest.raises(ValueError, match="contains unknown fields"):
        QueueControlConfig.from_document({"queue_control": {"stage_backpressure": unknown}}, num_stages=3)

    with pytest.raises(ValueError, match="must be a JSON object"):
        QueueControlConfig.from_document({"queue_control": {"stage_backpressure": None}}, num_stages=3)

    for unavailable in (_stage_backpressure_mapping(downstream_stage_ids=[1, 3]),):
        with pytest.raises(ValueError, match="stage_backpressure contains unavailable stage ids"):
            QueueControlConfig.from_document(
                {"queue_control": {"stage_backpressure": unavailable}},
                num_stages=3,
            )

    invalid_runtime_config = QueueControlConfig(
        enabled=True,
        stage_class_wip_limits={0: {"speech": 1}},
        stage_backpressure=_stage_backpressure(downstream_stage_ids=(1, 3)),
    )
    with pytest.raises(ValueError, match="stage_backpressure contains unavailable stage ids"):
        RuntimeQueueController(num_stages=3, config=invalid_runtime_config)

    controller = RuntimeQueueController(num_stages=3)
    with pytest.raises(ValueError, match="stage_backpressure contains unavailable stage ids"):
        controller.configure(invalid_runtime_config)


def test_enabled_stage_backpressure_requires_matching_upstream_stage_class_limit() -> None:
    with pytest.raises(ValueError, match="requires queue_control.enabled to be true"):
        QueueControlConfig(
            stage_class_wip_limits={0: {"speech": 1}},
            stage_backpressure=_stage_backpressure(),
        )
    with pytest.raises(ValueError, match="requires a matching queue_control.stage_class_wip_limits entry"):
        QueueControlConfig(enabled=True, stage_backpressure=_stage_backpressure())
    with pytest.raises(ValueError, match="requires a matching queue_control.stage_class_wip_limits entry"):
        QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"text": 1}},
            stage_backpressure=_stage_backpressure(),
        )

    config = QueueControlConfig(
        enabled=True,
        stage_class_wip_limits={0: {"speech": 0}},
        stage_backpressure=_stage_backpressure(),
    )
    assert config.stage_class_wip_limits == {0: {"speech": 0}}


def test_stage_backpressure_schema_v1_rejects_non_shared_upstream_deadlock_topology() -> None:
    with pytest.raises(ValueError, match="upstream_stage_id must be 0 in schema_version 1"):
        StageBackpressureConfig(
            enabled=True,
            upstream_stage_id=1,
            downstream_stage_ids=(2,),
            request_class="speech",
        )


def test_stage_backpressure_absence_preserves_legacy_fingerprint() -> None:
    legacy = QueueControlConfig()
    assert "stage_backpressure" not in legacy.semantic_mapping()
    assert legacy.fingerprint() == "2ab535c52658be3a4b35561ffafa999d900d1acd80a0ae2c5ef5a9a41fc4836c"
    controller = RuntimeQueueController(num_stages=3, config=legacy)
    assert not controller.active
    assert not controller.requires_queue(_pending("legacy", request_class="speech"))

    explicit_disabled = QueueControlConfig(stage_backpressure=_stage_backpressure(enabled=False))
    assert explicit_disabled.fingerprint() != legacy.fingerprint()


def test_stage_backpressure_presence_is_covered_by_allocator_cas() -> None:
    target = QueueControlConfig(
        enabled=True,
        stage_class_wip_limits={0: {"speech": 2}},
        stage_backpressure=_stage_backpressure(),
    )
    document = {"queue_control": target.semantic_mapping()}
    document["queue_control"]["online_allocator"] = {
        "schema_version": 2,
        "revision": 1,
        "source_runtime_id": "runtime-a",
        "source_snapshot_sequence": 1,
        "source_config_generation": 0,
        "source_config_fingerprint": "a" * 64,
        "target_config_fingerprint": target.fingerprint(),
        "profile_fingerprint": "b" * 64,
    }
    assert QueueControlConfig.from_document(document, num_stages=3).fingerprint() == target.fingerprint()

    document["queue_control"]["stage_backpressure"]["enabled"] = False
    with pytest.raises(ValueError, match="target_config_fingerprint does not match"):
        QueueControlConfig.from_document(document, num_stages=3)


def test_online_allocator_metadata_is_parsed_and_acknowledged_in_snapshot() -> None:
    target = QueueControlConfig(
        enabled=True,
        class_wip_limits={"audio": 3, "text": 1},
    )
    metadata = {
        "schema_version": 2,
        "revision": 7,
        "source_runtime_id": "runtime-a",
        "source_snapshot_sequence": 13,
        "source_config_generation": 2,
        "source_config_fingerprint": "b" * 64,
        "target_config_fingerprint": target.fingerprint(),
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
        source_config_fingerprint="b" * 64,
        target_config_fingerprint=target.fingerprint(),
        profile_fingerprint="a" * 64,
    )

    controller = RuntimeQueueController(num_stages=1, config=config)
    assert controller.snapshot()["online_allocator"] == metadata
    assert controller.snapshot()["queue_control_config_fingerprint"] == target.fingerprint()


def test_online_allocator_target_fingerprint_covers_full_semantic_config() -> None:
    target = QueueControlConfig(
        enabled=True,
        policy="edf",
        stage_class_wip_limits={0: {"interactive": 2}},
        admission=_admission_config(
            effective_k=2,
            service_samples_s=(0.1, 0.2),
        ).admission,
    )
    document = {"queue_control": target.semantic_mapping()}
    document["queue_control"]["online_allocator"] = {
        "schema_version": 2,
        "revision": 1,
        "source_runtime_id": "runtime-a",
        "source_snapshot_sequence": 1,
        "source_config_generation": 0,
        "source_config_fingerprint": "a" * 64,
        "target_config_fingerprint": target.fingerprint(),
        "profile_fingerprint": "b" * 64,
    }
    assert QueueControlConfig.from_document(document).fingerprint() == target.fingerprint()

    document["queue_control"]["admission"]["classes"]["interactive"]["gamma"] = 0.25
    with pytest.raises(ValueError, match="target_config_fingerprint does not match"):
        QueueControlConfig.from_document(document)


def test_online_allocator_revision_prevents_stale_or_torn_logical_updates() -> None:
    def config(revision: int, audio_limit: int) -> QueueControlConfig:
        target = QueueControlConfig(
            enabled=True,
            class_wip_limits={"audio": audio_limit},
        )
        return QueueControlConfig(
            enabled=True,
            class_wip_limits={"audio": audio_limit},
            online_allocator=OnlineAllocatorMetadata(
                revision=revision,
                source_runtime_id="runtime-a",
                source_snapshot_sequence=revision,
                source_config_generation=revision - 1,
                source_config_fingerprint="a" * 64,
                target_config_fingerprint=target.fingerprint(),
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
    replayed_target = QueueControlConfig(enabled=True, class_wip_limits={"audio": 1})
    replayed_source = QueueControlConfig(
        enabled=replayed_target.enabled,
        class_wip_limits=replayed_target.class_wip_limits,
        online_allocator=OnlineAllocatorMetadata(
            revision=4,
            source_runtime_id="runtime-a",
            source_snapshot_sequence=3,
            source_config_generation=3,
            source_config_fingerprint="a" * 64,
            target_config_fingerprint=replayed_target.fingerprint(),
            profile_fingerprint="b" * 64,
        ),
    )
    with pytest.raises(ValueError, match="source_snapshot_sequence must advance"):
        controller.configure(replayed_source)
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
    assert config.admission.enforce
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

    shadow = QueueControlConfig.from_document(
        {
            "queue_control": {
                "admission": {
                    "enabled": True,
                    "enforce": False,
                    "classes": {
                        "interactive": {
                            "effective_k": 1,
                            "mu": 1.0,
                            "service_samples_s": [0.5],
                            "gamma": 0.9,
                        }
                    },
                }
            }
        }
    )
    assert shadow.policy == "fifo"
    assert shadow.admission.enabled
    assert not shadow.admission.enforce

    with pytest.raises(ValueError, match="admission.enforce must be boolean"):
        QueueControlConfig.from_document(
            {
                "queue_control": {
                    "admission": {
                        "enforce": "false",
                        "classes": {
                            "interactive": {
                                "effective_k": 1,
                                "mu": 1.0,
                                "service_samples_s": [0.5],
                                "gamma": 0.9,
                            }
                        },
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


def test_vectorized_erlang_empirical_score_matches_scalar_reference_randomized() -> None:
    rng = random.Random(18401)
    for _ in range(500):
        effective_k = rng.randint(1, 16)
        active_count = rng.randint(0, effective_k + 1)
        queue_position = rng.randint(0, 128)
        remaining_budget_s = rng.uniform(-0.1, 20.0)
        mu = rng.uniform(0.01, 20.0)
        samples = tuple(rng.uniform(0.0, 10.0) for _ in range(rng.randint(1, 128)))
        vectorized = erlang_empirical_admission_score(
            remaining_budget_s,
            effective_k=effective_k,
            mu=mu,
            active_count=active_count,
            queue_position=queue_position,
            service_samples_s=samples,
        )
        scalar = erlang_empirical_admission_score_reference(
            remaining_budget_s,
            effective_k=effective_k,
            mu=mu,
            active_count=active_count,
            queue_position=queue_position,
            service_samples_s=samples,
        )
        assert vectorized == pytest.approx(scalar, abs=1e-12, rel=1e-12)


def _authoritative_admission_score(
    remaining_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    required_returns: int,
    service_samples_s: tuple[float, ...],
    gamma: float,
) -> float:
    if required_returns == 0:
        active_count = 0
        queue_position = 0
    else:
        active_count = effective_k
        queue_position = required_returns - 1
    score = erlang_empirical_admission_score(
        remaining_budget_s,
        effective_k=effective_k,
        mu=mu,
        active_count=active_count,
        queue_position=queue_position,
        service_samples_s=service_samples_s,
    )
    if abs(score - gamma) <= queue_control_module.ADMISSION_SCORE_REFERENCE_TOLERANCE:
        score = erlang_empirical_admission_score_reference(
            remaining_budget_s,
            effective_k=effective_k,
            mu=mu,
            active_count=active_count,
            queue_position=queue_position,
            service_samples_s=service_samples_s,
        )
    return score


def _vectorized_admission_score(
    remaining_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    required_returns: int,
    service_samples_s: tuple[float, ...],
) -> float:
    if required_returns == 0:
        active_count = 0
        queue_position = 0
    else:
        active_count = effective_k
        queue_position = required_returns - 1
    return erlang_empirical_admission_score(
        remaining_budget_s,
        effective_k=effective_k,
        mu=mu,
        active_count=active_count,
        queue_position=queue_position,
        service_samples_s=service_samples_s,
    )


@pytest.mark.parametrize("gamma", [0.0, 0.65])
def test_compiled_admission_thresholds_match_vectorized_predicate(gamma: float) -> None:
    samples = (0.05, 0.2, 0.4, 0.9)
    thresholds = compile_erlang_empirical_admission_thresholds(
        effective_k=3,
        mu=0.7,
        service_samples_s=samples,
        gamma=gamma,
        max_required_returns=8,
    )
    assert len(thresholds) == 9
    assert list(thresholds) == sorted(thresholds)

    rng = random.Random(18403 + int(gamma * 100))
    for required_returns, threshold in enumerate(thresholds):
        boundary_target = (
            gamma
            if required_returns == 0 or gamma == 0.0
            else gamma + queue_control_module.ADMISSION_SCORE_REFERENCE_TOLERANCE
        )
        assert (
            _vectorized_admission_score(
                threshold,
                effective_k=3,
                mu=0.7,
                required_returns=required_returns,
                service_samples_s=samples,
            )
            >= boundary_target
        )
        if threshold > 0.0:
            previous_score = _vectorized_admission_score(
                math.nextafter(threshold, -math.inf),
                effective_k=3,
                mu=0.7,
                required_returns=required_returns,
                service_samples_s=samples,
            )
            if required_returns == 0:
                assert previous_score < gamma
            else:
                assert previous_score <= boundary_target
        for _ in range(8):
            remaining_budget_s = max(0.0, threshold + rng.uniform(-0.5, 0.5))
            expected = (
                _authoritative_admission_score(
                    remaining_budget_s,
                    effective_k=3,
                    mu=0.7,
                    required_returns=required_returns,
                    service_samples_s=samples,
                    gamma=gamma,
                )
                >= gamma
            )
            assert (remaining_budget_s >= threshold) == expected


def test_zero_return_threshold_handles_float_product_boundary() -> None:
    gamma = math.nextafter(1.0 / 3.0, math.inf)
    thresholds = compile_erlang_empirical_admission_thresholds(
        effective_k=1,
        mu=1.0,
        service_samples_s=(0.1, 0.2, 0.3),
        gamma=gamma,
        max_required_returns=0,
    )
    assert thresholds == (0.2,)


def test_threshold_tie_guard_is_a_conservative_subset_of_legacy_predicate() -> None:
    samples = (0.05, 0.2, 0.4, 0.9)
    gamma = 0.65
    compiled_threshold = compile_erlang_empirical_admission_thresholds(
        effective_k=3,
        mu=0.7,
        service_samples_s=samples,
        gamma=gamma,
        max_required_returns=1,
    )[1]

    low = 0.0
    high = compiled_threshold
    while True:
        midpoint = low + (high - low) / 2.0
        if midpoint == low or midpoint == high:
            break
        score = _authoritative_admission_score(
            midpoint,
            effective_k=3,
            mu=0.7,
            required_returns=1,
            service_samples_s=samples,
            gamma=gamma,
        )
        if score >= gamma:
            high = midpoint
        else:
            low = midpoint
    legacy_threshold = high
    assert legacy_threshold < compiled_threshold

    tie_budget = legacy_threshold + (compiled_threshold - legacy_threshold) / 2.0
    assert tie_budget < compiled_threshold
    assert (
        _authoritative_admission_score(
            tie_budget,
            effective_k=3,
            mu=0.7,
            required_returns=1,
            service_samples_s=samples,
            gamma=gamma,
        )
        >= gamma
    )


def test_threshold_compiler_rejects_gamma_one() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        compile_erlang_empirical_admission_thresholds(
            effective_k=1,
            mu=1.0,
            service_samples_s=(0.1,),
            gamma=1.0,
            max_required_returns=1,
        )


def test_threshold_compiler_rejects_unrepresentable_finite_boundary() -> None:
    with pytest.raises(ValueError, match="no representable finite threshold"):
        compile_erlang_empirical_admission_thresholds(
            effective_k=1,
            mu=1e-308,
            service_samples_s=(1e308,),
            gamma=0.5,
            max_required_returns=1,
        )


def test_threshold_control_requires_and_validates_bound_artifact() -> None:
    table = compile_erlang_empirical_admission_threshold_table(
        "interactive",
        effective_k=2,
        mu=1.25,
        service_samples_s=(0.1, 0.3, 0.7),
        gamma=0.8,
        max_required_returns=4,
    )
    class_document = {
        "effective_k": 2,
        "mu": 1.25,
        "service_samples_s": [0.1, 0.3, 0.7],
        "gamma": 0.8,
        **table.to_config_fields(),
    }
    document = {
        "queue_control": {
            "policy": "edf",
            "admission": {
                "enabled": True,
                "score_method": "erlang_empirical_threshold",
                "classes": {"interactive": class_document},
            },
        }
    }
    config = QueueControlConfig.from_document(document)
    installed = config.admission.threshold_tables["interactive"]
    assert installed == table
    assert installed.max_required_returns == 4
    semantic_class = config.semantic_mapping()["admission"]["classes"]["interactive"]
    assert "compiled_thresholds_s" not in semantic_class
    assert semantic_class["max_required_returns"] == 4
    assert semantic_class["threshold_profile_fingerprint"] == table.profile_fingerprint
    assert semantic_class["threshold_table_digest"] == table.table_digest

    missing = json.loads(json.dumps(document))
    del missing["queue_control"]["admission"]["classes"]["interactive"]["compiled_thresholds_s"]
    with pytest.raises(ValueError, match="must be provided together"):
        QueueControlConfig.from_document(missing)

    stale_profile = json.loads(json.dumps(document))
    stale_profile["queue_control"]["admission"]["classes"]["interactive"]["mu"] = 2.0
    with pytest.raises(ValueError, match="profile fingerprint does not match"):
        QueueControlConfig.from_document(stale_profile)

    corrupt_table = json.loads(json.dumps(document))
    corrupt_table["queue_control"]["admission"]["classes"]["interactive"]["compiled_thresholds_s"][-1] += 0.1
    with pytest.raises(ValueError, match="table digest does not match"):
        QueueControlConfig.from_document(corrupt_table)

    forged_zero_table = json.loads(json.dumps(document))
    forged_class = forged_zero_table["queue_control"]["admission"]["classes"]["interactive"]
    forged_class["compiled_thresholds_s"] = [0.0] * len(forged_class["compiled_thresholds_s"])
    forged_class["threshold_table_digest"] = admission_threshold_table_digest(
        profile_fingerprint=table.profile_fingerprint,
        thresholds_s=forged_class["compiled_thresholds_s"],
    )
    with pytest.raises(ValueError, match="does not pass its predicate"):
        QueueControlConfig.from_document(forged_zero_table)

    alternate_table = compile_erlang_empirical_admission_threshold_table(
        "interactive",
        effective_k=2,
        mu=1.25,
        service_samples_s=(0.1, 0.3, 0.7),
        gamma=0.7,
        max_required_returns=4,
    )
    alternate = json.loads(json.dumps(document))
    alternate_class = alternate["queue_control"]["admission"]["classes"]["interactive"]
    alternate_class["gamma"] = 0.7
    alternate_class.update(alternate_table.to_config_fields())
    alternate_config = QueueControlConfig.from_document(alternate)
    assert alternate_config != config
    assert alternate_config.fingerprint() != config.fingerprint()
    controller = RuntimeQueueController(num_stages=1, config=config)
    assert controller.configure(alternate_config)
    assert controller.config_generation == 1
    assert (
        controller.config.admission.threshold_tables["interactive"].table_digest
        == alternate_class["threshold_table_digest"]
    )

    with pytest.raises(ValueError, match="require score_method"):
        AdmissionControlConfig(
            enabled=True,
            classes={
                "interactive": AdmissionClassConfig(
                    effective_k=2,
                    mu=1.25,
                    service_samples_s=(0.1, 0.3, 0.7),
                    gamma=0.8,
                    max_required_returns=4,
                    compiled_thresholds_s=table.thresholds_s,
                    threshold_profile_fingerprint=table.profile_fingerprint,
                    threshold_table_digest=table.table_digest,
                )
            },
        )


def test_threshold_profile_and_table_digests_cover_all_inputs() -> None:
    base = admission_threshold_profile_fingerprint(
        "interactive",
        effective_k=2,
        mu=1.0,
        service_samples_s=(0.1, 0.2),
        gamma=0.8,
        max_required_returns=4,
    )
    assert base != admission_threshold_profile_fingerprint(
        "batch",
        effective_k=2,
        mu=1.0,
        service_samples_s=(0.1, 0.2),
        gamma=0.8,
        max_required_returns=4,
    )
    assert base != admission_threshold_profile_fingerprint(
        "interactive",
        effective_k=2,
        mu=1.0,
        service_samples_s=(0.1, 0.2),
        gamma=0.8,
        max_required_returns=5,
    )
    assert admission_threshold_table_digest(
        profile_fingerprint=base,
        thresholds_s=(0.1, 0.2),
    ) != admission_threshold_table_digest(
        profile_fingerprint=base,
        thresholds_s=(0.1, 0.3),
    )
    assert DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS >= 2048


def test_threshold_artifact_json_round_trip_canonicalizes_integer_floats() -> None:
    table = compile_erlang_empirical_admission_threshold_table(
        "interactive",
        effective_k=1,
        mu=1,
        service_samples_s=(0,),
        gamma=0,
        max_required_returns=1,
    )
    document = {
        "queue_control": {
            "policy": "edf",
            "admission": {
                "score_method": "erlang_empirical_threshold",
                "classes": {
                    "interactive": {
                        "effective_k": 1,
                        "mu": 1,
                        "service_samples_s": [0],
                        "gamma": 0,
                        **table.to_config_fields(),
                    }
                },
            },
        }
    }
    config = QueueControlConfig.from_document(json.loads(json.dumps(document)))
    assert config.admission.threshold_tables["interactive"] == table


def test_threshold_boundary_validation_is_cached() -> None:
    validator = queue_control_module._validate_erlang_empirical_admission_threshold_table_cached
    validator.cache_clear()
    table = compile_erlang_empirical_admission_threshold_table(
        "interactive",
        effective_k=2,
        mu=1.0,
        service_samples_s=(0.1, 0.2, 0.4),
        gamma=0.8,
        max_required_returns=4,
    )
    kwargs = {
        "request_class": "interactive",
        "effective_k": 2,
        "mu": 1.0,
        "service_samples_s": (0.1, 0.2, 0.4),
        "gamma": 0.8,
        "profile_fingerprint": table.profile_fingerprint,
        "table_digest": table.table_digest,
        "thresholds_s": table.thresholds_s,
    }
    queue_control_module.validate_erlang_empirical_admission_threshold_table(**kwargs)
    first = validator.cache_info()
    queue_control_module.validate_erlang_empirical_admission_threshold_table(**kwargs)
    second = validator.cache_info()
    assert first.misses == 1
    assert second.misses == 1
    assert second.hits == first.hits + 1


def test_request_metadata_builds_absolute_deadline() -> None:
    metadata = RequestSchedulingMetadata.create(
        path="audio",
        first_output_deadline_s=0.4,
        admission_correlation_id="client-request-7",
        now_monotonic_s=10.0,
    )
    assert metadata.request_class == "audio"
    assert metadata.path == "audio"
    assert metadata.deadline_monotonic_s == pytest.approx(10.4)
    assert metadata.admission_correlation_id == "client-request-7"

    anchored = RequestSchedulingMetadata.create(
        first_output_deadline_monotonic_s=10.4,
        now_monotonic_s=99.0,
    )
    assert anchored.deadline_monotonic_s == 10.4

    with pytest.raises(ValueError, match="non-negative"):
        RequestSchedulingMetadata.create(first_output_deadline_s=-0.1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RequestSchedulingMetadata.create(
            first_output_deadline_s=0.4,
            first_output_deadline_monotonic_s=10.4,
        )


def test_http_headers_require_explicit_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = {
        "X-VLLM-OMNI-REQUEST-CLASS": "interactive",
        "x-vllm-omni-request-path": "audio",
        "x-vllm-omni-first-output-deadline-ms": "400",
        "x-vllm-omni-admission-correlation-id": "client-request-7",
    }
    assert scheduling_kwargs_from_headers(headers) == {}

    monkeypatch.setenv("VLLM_OMNI_TRUST_SCHEDULING_HEADERS", "1")
    assert scheduling_kwargs_from_headers(headers) == {
        "request_class": "interactive",
        "request_path": "audio",
        "first_output_deadline_s": 0.4,
        "admission_correlation_id": "client-request-7",
    }
    assert scheduling_kwargs_from_headers(
        headers,
        deadline_anchor_monotonic_s=10.0,
    ) == {
        "request_class": "interactive",
        "request_path": "audio",
        "first_output_deadline_monotonic_s": 10.4,
        "admission_correlation_id": "client-request-7",
    }

    with pytest.raises(ValueError, match="finite and non-negative"):
        scheduling_kwargs_from_headers(
            {"x-vllm-omni-first-output-deadline-ms": "-1"},
            trusted=True,
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        scheduling_kwargs_from_headers(
            {"x-vllm-omni-admission-correlation-id": "  "},
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


def test_stage0_multimodal_cache_order_fence_prevents_cross_class_bypass() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            policy="edf",
            stage_class_wip_limits={0: {"text": 1, "speech": 2}},
        ),
    )
    controller.enqueue(
        _pending(
            "text-active",
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    assert controller.pop_ready().pending.request_id == "text-active"  # type: ignore[union-attr]

    # P0 processed producer before consumer.  The producer is blocked by its
    # text credit while EDF would otherwise dispatch the speech consumer first.
    controller.enqueue(
        _pending(
            "producer",
            deadline=20.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    consumer = _pending(
        "consumer",
        deadline=10.0,
        request_class="speech",
        preserve_stage0_mm_cache_order=True,
    )
    assert controller.requires_queue(consumer)
    controller.enqueue(consumer)
    assert controller.pop_ready() is None
    assert controller.snapshot()["blocked_by_limit"] == {
        "mm_cache_order": 1,
        "stage_class": 1,
    }

    # Text-only work does not touch the P0/P1 cache and may still use a free
    # speech credit while the multimodal fence waits.
    controller.enqueue(
        _pending(
            "text-only",
            deadline=5.0,
            request_class="speech",
        )
    )
    assert controller.pop_ready().pending.request_id == "text-only"  # type: ignore[union-attr]

    assert controller.release_stage("text-active", 0)
    assert controller.pop_ready().pending.request_id == "producer"  # type: ignore[union-attr]
    assert controller.pop_ready().pending.request_id == "consumer"  # type: ignore[union-attr]


def test_stage0_multimodal_cache_order_fence_survives_live_disable() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            policy="edf",
            stage_class_wip_limits={0: {"text": 0}},
        ),
    )
    controller.enqueue(
        _pending(
            "producer",
            deadline=20.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    assert controller.pop_ready() is None

    controller.configure(QueueControlConfig(enabled=False, policy="edf"))
    consumer = _pending(
        "consumer",
        deadline=10.0,
        request_class="speech",
        preserve_stage0_mm_cache_order=True,
    )
    assert controller.requires_queue(consumer)
    controller.enqueue(consumer)

    assert controller.pop_ready().pending.request_id == "producer"  # type: ignore[union-attr]
    assert controller.pop_ready().pending.request_id == "consumer"  # type: ignore[union-attr]


def test_soft_stage_class_reservation_borrows_idle_share_up_to_stage_cap() -> None:
    controller = RuntimeQueueController(num_stages=1, config=_soft_stage0_config())

    controller.enqueue(_pending("text-reserved", request_class="text"))
    assert controller.pop_ready().pending.request_id == "text-reserved"  # type: ignore[union-attr]

    controller.enqueue(_pending("text-borrowed", request_class="text"))
    assert controller.pop_ready().pending.request_id == "text-borrowed"  # type: ignore[union-attr]

    controller.enqueue(_pending("text-blocked", request_class="text"))
    assert controller.pop_ready() is None
    snapshot = controller.snapshot()
    assert snapshot["blocked_by_limit"] == {"stage": 1}
    assert snapshot["soft_reservation_state"]["0"] == {
        "stage_wip_limit": 2,
        "reservations": {"speech": 1, "text": 1},
        "active_total": 2,
        "active_by_class": {"speech": 0, "text": 2},
        "demand_classes": [],
        "reservation_priority_dispatch_total": 0,
        "borrowed_dispatch_total": 1,
        "contended_borrowed_dispatch_total": 0,
        "cache_order_head_exempt_dispatch_total": 0,
        "global_cap_block_events_total": 1,
        "global_cap_blocked_pending": 1,
    }


def test_soft_stage_class_reservation_prioritizes_an_underfilled_class() -> None:
    controller = RuntimeQueueController(num_stages=1, config=_soft_stage0_config())
    controller.enqueue(_pending("text-active", request_class="text"))
    assert controller.pop_ready().pending.request_id == "text-active"  # type: ignore[union-attr]

    # FIFO would choose text-borrower first. The speech reservation has ready
    # demand, so speech receives its reserved slot before text may borrow it.
    controller.enqueue(_pending("text-borrower", request_class="text"))
    controller.enqueue(_pending("speech-reserved", request_class="speech"))
    snapshot = controller.snapshot()
    assert snapshot["blocked_by_limit"] == {"stage_class_reservation": 1}
    assert snapshot["soft_reservation_state"]["0"]["demand_classes"] == ["speech"]

    assert controller.pop_ready().pending.request_id == "speech-reserved"  # type: ignore[union-attr]
    snapshot = controller.snapshot()
    assert snapshot["soft_reservation_state"]["0"]["reservation_priority_dispatch_total"] == 1
    assert snapshot["soft_reservation_state"]["0"]["active_by_class"] == {
        "speech": 1,
        "text": 1,
    }


def test_soft_stage_class_reservation_tracks_cache_blocked_demand_and_advances_head() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_soft_stage0_config(policy="edf"),
    )
    controller.enqueue(
        _pending(
            "text-active",
            deadline=30.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    assert controller.pop_ready().pending.request_id == "text-active"  # type: ignore[union-attr]

    # The later speech request has an earlier deadline and an unused reserved
    # share. It cannot bypass the multimodal cache head, but its latent demand
    # prevents unrelated borrowers from consuming the reservation. The head is
    # exempt so that the cache-ordered sequence can still advance.
    controller.enqueue(
        _pending(
            "cache-head",
            deadline=20.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    controller.enqueue(
        _pending(
            "cache-follower",
            deadline=10.0,
            request_class="speech",
            preserve_stage0_mm_cache_order=True,
        )
    )
    snapshot = controller.snapshot()
    assert snapshot["blocked_by_limit"] == {"mm_cache_order": 1}
    assert snapshot["soft_reservation_state"]["0"]["demand_classes"] == ["speech"]
    assert controller.pop_ready().pending.request_id == "cache-head"  # type: ignore[union-attr]

    assert controller.release_stage("text-active", 0)
    assert controller.pop_ready().pending.request_id == "cache-follower"  # type: ignore[union-attr]


def test_soft_stage_class_reservation_blocks_arbitrary_borrower_behind_cache_head() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_soft_stage0_config(policy="edf"),
    )
    controller.enqueue(_pending("speech-active", request_class="speech"))
    assert controller.pop_ready().pending.request_id == "speech-active"  # type: ignore[union-attr]

    controller.enqueue(
        _pending(
            "cache-head",
            deadline=30.0,
            request_class="speech",
            preserve_stage0_mm_cache_order=True,
        )
    )
    controller.enqueue(
        _pending(
            "cache-follower",
            deadline=10.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    controller.enqueue(_pending("arbitrary-borrower", deadline=5.0, request_class="speech"))

    snapshot = controller.snapshot()
    assert snapshot["soft_reservation_state"]["0"]["demand_classes"] == ["text"]
    assert snapshot["blocked_by_limit"] == {
        "mm_cache_order": 1,
        "stage_class_reservation": 1,
    }
    assert controller.pop_ready().pending.request_id == "cache-head"  # type: ignore[union-attr]
    snapshot = controller.snapshot()
    assert snapshot["soft_reservation_state"]["0"]["active_total"] == 2
    assert snapshot["soft_reservation_state"]["0"]["demand_classes"] == ["text"]

    assert controller.release_stage("speech-active", 0)
    assert controller.pop_ready().pending.request_id == "cache-follower"  # type: ignore[union-attr]
    snapshot = controller.snapshot()
    assert snapshot["soft_reservation_state"]["0"]["active_total"] == 2
    assert snapshot["soft_reservation_state"]["0"]["active_by_class"] == {
        "speech": 1,
        "text": 1,
    }
    assert snapshot["soft_reservation_state"]["0"]["borrowed_dispatch_total"] == 1
    assert (
        snapshot["soft_reservation_state"]["0"][
            "contended_borrowed_dispatch_total"
        ]
        == 1
    )
    assert (
        snapshot["soft_reservation_state"]["0"][
            "cache_order_head_exempt_dispatch_total"
        ]
        == 1
    )
    assert snapshot["queued_by_stage_class"] == {"0": {"speech": 1}}


def test_soft_stage_class_reservation_does_not_promote_multiply_blocked_cache_demand() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=_soft_stage0_config(policy="edf"),
    )
    controller.enqueue(_pending("text-active", request_class="text"))
    assert controller.pop_ready().pending.request_id == "text-active"  # type: ignore[union-attr]

    controller.enqueue(
        _pending(
            "cache-head",
            deadline=20.0,
            request_class="text",
            preserve_stage0_mm_cache_order=True,
        )
    )
    controller.enqueue(
        _pending(
            "cache-follower",
            deadline=10.0,
            request_class="speech",
            required_active_stage_id=1,
            preserve_stage0_mm_cache_order=True,
        )
    )

    snapshot = controller.snapshot()
    assert snapshot["soft_reservation_state"]["0"]["demand_classes"] == []
    assert snapshot["blocked_by_limit"] == {
        "dependency": 1,
        "mm_cache_order": 1,
    }
    assert controller.pop_ready().pending.request_id == "cache-head"  # type: ignore[union-attr]
    assert controller.snapshot()["soft_reservation_state"]["0"]["active_total"] == 2


def test_soft_stage_class_reservation_can_be_enabled_by_live_config() -> None:
    hard = QueueControlConfig(
        enabled=True,
        stage_wip_limits={0: 2},
        stage_class_wip_limits={0: {"text": 1, "speech": 1}},
    )
    controller = RuntimeQueueController(num_stages=1, config=hard)
    controller.enqueue(_pending("text-active", request_class="text"))
    assert controller.pop_ready().pending.request_id == "text-active"  # type: ignore[union-attr]
    controller.enqueue(_pending("text-waiting", request_class="text"))
    assert controller.pop_ready() is None

    assert controller.configure(_soft_stage0_config())
    assert controller.pop_ready().pending.request_id == "text-waiting"  # type: ignore[union-attr]
    snapshot = controller.snapshot()
    assert snapshot["stage_class_wip_modes"] == {"0": "soft_reservation"}
    assert snapshot["soft_reservation_state"]["0"]["borrowed_dispatch_total"] == 1


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


def test_disabled_stage_backpressure_is_auditable_but_does_not_queue() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_backpressure=_stage_backpressure(enabled=False),
        ),
    )
    controller.acquire_immediate(_pending("downstream", stage_id=1, starts_request=False, request_class="speech"))
    upstream = _pending("upstream", request_class="speech")

    assert not controller.requires_queue(upstream)
    controller.acquire_immediate(upstream)
    snapshot = controller.snapshot()
    assert snapshot["stage_backpressure"] == _stage_backpressure_mapping(enabled=False)
    assert snapshot["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 1,
        "downstream_unfinished_logical": 1,
        "overlap_unfinished_logical": 0,
        "blocking": False,
    }
    assert snapshot["blocked_by_limit"] == {}


def test_stage_backpressure_blocks_and_unblocks_matching_upstream_dispatch() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    assert controller.snapshot()["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 0,
        "downstream_unfinished_logical": 0,
        "overlap_unfinished_logical": 0,
        "blocking": False,
    }
    controller.acquire_immediate(_pending("downstream", stage_id=1, starts_request=False, request_class="speech"))
    upstream = _pending("upstream", request_class="speech")

    assert controller.requires_queue(upstream)
    controller.enqueue(upstream)
    assert controller.pop_ready() is None
    snapshot = controller.snapshot()
    assert snapshot["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 1,
        "downstream_unfinished_logical": 1,
        "overlap_unfinished_logical": 0,
        "blocking": True,
    }
    assert snapshot["blocked_by_limit"] == {"backpressure": 1}

    assert controller.release_stage("downstream", 1)
    snapshot = controller.snapshot()
    assert snapshot["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 1,
        "downstream_unfinished_logical": 0,
        "overlap_unfinished_logical": 0,
        "blocking": False,
    }
    acquired = controller.pop_ready()
    assert acquired is not None and acquired.pending.request_id == "upstream"


def test_stage_backpressure_allows_existing_lease_updates_while_blocking_new_work() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    controller.acquire_immediate(_pending("upstream-active", request_class="speech"))
    controller.acquire_immediate(_pending("downstream-1", stage_id=1, starts_request=False, request_class="speech"))
    controller.acquire_immediate(_pending("downstream-2", stage_id=2, starts_request=False, request_class="speech"))

    update = _pending("upstream-active", starts_request=False, request_class="speech")
    new_request = _pending("upstream-new", request_class="speech")
    assert controller.requires_queue(update)
    assert controller.requires_queue(new_request)
    controller.enqueue(update)
    controller.enqueue(new_request)

    snapshot = controller.snapshot()
    assert snapshot["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 2,
        "downstream_unfinished_logical": 2,
        "overlap_unfinished_logical": 0,
        "blocking": True,
    }
    assert snapshot["blocked_by_limit"] == {"backpressure": 1}
    acquired = controller.pop_ready()
    assert acquired is not None and acquired.pending.request_id == "upstream-active"
    assert not acquired.acquired_stage
    assert controller.pop_ready() is None

    assert controller.release_stage("downstream-2", 2)
    acquired = controller.pop_ready()
    assert acquired is not None and acquired.pending.request_id == "upstream-new"


def test_stage_backpressure_drains_mm_cache_predecessor_before_existing_lease_update() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    controller.acquire_immediate(_pending("streaming", request_class="speech"))
    controller.acquire_immediate(_pending("downstream-1", stage_id=1, starts_request=False, request_class="speech"))
    controller.acquire_immediate(_pending("downstream-2", stage_id=2, starts_request=False, request_class="speech"))
    predecessor = _pending(
        "new-request",
        request_class="speech",
        preserve_stage0_mm_cache_order=True,
    )
    existing_lease_update = _pending(
        "streaming",
        starts_request=False,
        request_class="speech",
        preserve_stage0_mm_cache_order=True,
    )
    controller.enqueue(predecessor)
    controller.enqueue(existing_lease_update)

    snapshot = controller.snapshot()
    assert snapshot["stage_backpressure_state"]["blocking"] is True
    assert snapshot["blocked_by_limit"] == {"mm_cache_order": 1}
    first = controller.pop_ready()
    second = controller.pop_ready()
    assert first is not None and first.pending.request_id == "new-request"
    assert second is not None and second.pending.request_id == "streaming"
    assert not second.acquired_stage


def test_stage_backpressure_deduplicates_downstream_stages_and_reports_overlap() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    controller.acquire_immediate(_pending("overlap", request_class="speech"))
    controller.acquire_immediate(_pending("overlap", stage_id=1, starts_request=False, request_class="speech"))
    controller.acquire_immediate(_pending("overlap", stage_id=2, starts_request=False, request_class="speech"))
    controller.enqueue(_pending("overlap", stage_id=2, starts_request=False, request_class="speech"))

    assert controller.snapshot()["stage_backpressure_state"] == {
        "schema_version": 1,
        "upstream_unfinished_logical": 1,
        "downstream_unfinished_logical": 1,
        "overlap_unfinished_logical": 1,
        "blocking": True,
    }


def test_stage_backpressure_ignores_non_target_class_and_stage() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )

    assert not controller.requires_queue(_pending("text-upstream", request_class="text"))
    assert not controller.requires_queue(
        _pending("speech-downstream", stage_id=1, starts_request=True, request_class="speech")
    )
    assert controller.requires_queue(_pending("speech-upstream", request_class="speech"))


def test_unfinished_logical_state_is_computed_once_per_pop_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 100}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    controller.acquire_immediate(_pending("downstream", stage_id=1, starts_request=False, request_class="speech"))
    controller.enqueue(_pending("upstream-1", request_class="speech"))
    controller.enqueue(_pending("upstream-2", request_class="speech"))

    calls = 0
    original = controller._unfinished_logical_ids_by_stage_class

    def counted_state() -> dict[tuple[int, str], set[str]]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(controller, "_unfinished_logical_ids_by_stage_class", counted_state)
    assert controller.pop_ready() is not None
    assert calls == 1

    calls = 0
    controller.snapshot()
    assert calls == 1


def test_disabled_stage_backpressure_skips_pop_state_but_keeps_snapshot_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_backpressure=_stage_backpressure(enabled=False),
        ),
    )
    controller.enqueue(_pending("upstream", request_class="speech"))

    calls = 0
    original = controller._unfinished_logical_ids_by_stage_class

    def counted_state() -> dict[tuple[int, str], set[str]]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(controller, "_unfinished_logical_ids_by_stage_class", counted_state)
    assert controller.pop_ready() is not None
    assert calls == 0

    controller.snapshot()
    assert calls == 1


def test_block_evaluation_precomputes_request_counts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            path_wip_limits={"speech": 1},
            class_wip_limits={"speech": 1},
        ),
    )
    controller.acquire_immediate(_pending("active", request_class="speech", path="speech"))
    for index in range(20):
        controller.enqueue(_pending(f"pending-{index}", request_class="speech", path="speech"))

    calls = 0
    original = controller._request_counts

    def counted_request_counts() -> tuple[object, object]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(controller, "_request_counts", counted_request_counts)
    assert controller.pop_ready() is None
    assert calls == 1

    calls = 0
    controller.snapshot()
    assert calls == 1


def test_large_backpressure_queue_uses_only_linear_state_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingDict(dict[tuple[str, int], str]):
        items_calls = 0

        def items(self):  # type: ignore[no-untyped-def]
            self.items_calls += 1
            return super().items()

    class CountingList(list[PendingStageDispatch]):
        iter_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iter_calls += 1
            return super().__iter__()

    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"speech": 2000}},
            stage_backpressure=_stage_backpressure(),
        ),
    )
    for index in range(1000):
        controller.acquire_immediate(
            _pending(f"downstream-{index}", stage_id=1, starts_request=False, request_class="speech")
        )
        controller.enqueue(_pending(f"upstream-{index}", request_class="speech"))

    active_stage_classes = CountingDict(controller._active_stage_classes)
    pending = CountingList(controller._pending)
    controller._active_stage_classes = active_stage_classes
    controller._pending = pending

    def unexpected_request_counts() -> tuple[object, object]:
        raise AssertionError("request counts must be skipped when no path/class limits are configured")

    monkeypatch.setattr(controller, "_request_counts", unexpected_request_counts)
    assert controller.pop_ready() is None
    assert active_stage_classes.items_calls == 1
    assert pending.iter_calls == 2


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


def test_released_upstream_still_satisfies_a_queued_downstream_dependency() -> None:
    controller = RuntimeQueueController(
        num_stages=2,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 0}},
        ),
    )
    controller.enqueue(_pending("r1", request_class="speech"))
    assert controller.pop_ready().pending.stage_id == 0  # type: ignore[union-attr]

    controller.enqueue(
        _pending(
            "r1",
            stage_id=1,
            starts_request=False,
            request_class="speech",
            required_active_stage_id=0,
        )
    )
    assert controller.pop_ready() is None
    assert controller.release_stage("r1", 0)

    controller.configure(
        QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={1: {"speech": 1}},
        )
    )
    assert controller.pop_ready().pending.stage_id == 1  # type: ignore[union-attr]


def test_downstream_unfinished_logical_excludes_stage0_and_deduplicates_pending_stages() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(enabled=True),
    )
    controller.acquire_immediate(_pending("stage0-only", request_class="speech"))
    assert controller.snapshot()["downstream_unfinished_logical_by_class"] == {}

    controller.enqueue(
        _pending(
            "pending-stage-1",
            logical_request_id="pending-logical",
            stage_id=1,
            starts_request=False,
            request_class="speech",
        )
    )
    controller.enqueue(
        _pending(
            "pending-stage-2",
            logical_request_id="pending-logical",
            stage_id=2,
            starts_request=False,
            request_class="speech",
        )
    )
    assert controller.snapshot()["downstream_unfinished_logical_by_class"] == {"speech": 1}


def test_unfinished_logical_by_stage_class_deduplicates_active_and_pending_updates() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(enabled=True),
    )
    controller.acquire_immediate(_pending("speech-1", request_class="speech"))
    controller.enqueue(_pending("speech-1", starts_request=False, request_class="speech"))
    assert controller.snapshot()["unfinished_logical_by_stage_class"] == {"0": {"speech": 1}}

    controller.acquire_immediate(_pending("speech-2", request_class="speech"))
    controller.acquire_immediate(_pending("speech-1", stage_id=1, starts_request=False, request_class="speech"))
    controller.enqueue(_pending("text-1", request_class="text"))
    controller.enqueue(
        _pending(
            "text-stage-2",
            logical_request_id="text-1",
            stage_id=2,
            starts_request=False,
            request_class="text",
        )
    )
    assert controller.snapshot()["unfinished_logical_by_stage_class"] == {
        "0": {"speech": 2, "text": 1},
        "1": {"speech": 1},
        "2": {"text": 1},
    }


def test_downstream_unfinished_logical_count_deduplicates_stage_leases_and_queue() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={2: {"speech": 1}},
        ),
    )

    controller.acquire_immediate(_pending("r1", stage_id=1, starts_request=False, request_class="speech"))
    controller.acquire_immediate(_pending("r1", stage_id=2, starts_request=False, request_class="speech"))
    snapshot = controller.snapshot()
    assert snapshot["active_by_stage_class"] == {
        "1": {"speech": 1},
        "2": {"speech": 1},
    }
    assert snapshot["downstream_unfinished_logical_by_class"] == {"speech": 1}

    controller.acquire_immediate(_pending("r2", stage_id=1, starts_request=False, request_class="speech"))
    controller.enqueue(_pending("r2", stage_id=2, starts_request=False, request_class="speech"))
    snapshot = controller.snapshot()
    assert snapshot["queued_by_stage_class"] == {"2": {"speech": 1}}
    assert snapshot["downstream_unfinished_logical_by_class"] == {"speech": 2}

    controller.cancel_request("r1")
    assert controller.snapshot()["downstream_unfinished_logical_by_class"] == {"speech": 1}
    controller.cancel_request("r2")
    assert controller.snapshot()["downstream_unfinished_logical_by_class"] == {}


def test_dependency_completion_excludes_rollback_and_is_cleared_on_cancel() -> None:
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
    controller.enqueue(_pending("r1"))
    upstream = controller.pop_ready()
    assert upstream is not None and upstream.pending.stage_id == 0
    controller.rollback(upstream)
    assert controller.pop_ready() is None

    controller.enqueue(_pending("r1"))
    upstream = controller.pop_ready()
    assert upstream is not None and upstream.pending.stage_id == 0
    assert controller.release_stage("r1", 0)
    assert controller.pop_ready().pending.stage_id == 1  # type: ignore[union-attr]

    controller.cancel_request("r1")
    controller.acquire_immediate(_pending("r1", stage_id=2))
    controller.enqueue(
        _pending(
            "r1",
            stage_id=1,
            starts_request=False,
            required_active_stage_id=0,
        )
    )
    assert controller.pop_ready() is None


def test_parent_cancel_clears_completed_child_stage_after_later_rollback() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(enabled=True),
    )
    controller.acquire_immediate(_pending("parent"))
    child_stage = controller.acquire_immediate(
        _pending(
            "child",
            logical_request_id="parent",
            stage_id=1,
            starts_request=False,
        )
    )
    assert child_stage.acquired_stage
    assert controller.release_stage("child", 1)

    failed_child_stage = controller.acquire_immediate(
        _pending(
            "child",
            logical_request_id="parent",
            stage_id=2,
            starts_request=False,
        )
    )
    controller.rollback(failed_child_stage)
    controller.cancel_request("parent")

    controller.acquire_immediate(_pending("new-parent"))
    controller.enqueue(
        _pending(
            "child",
            logical_request_id="new-parent",
            stage_id=2,
            starts_request=False,
            required_active_stage_id=1,
        )
    )
    assert controller.pop_ready() is None


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
    assert snapshot["admission"]["actual_rejected_total"] == 1
    assert snapshot["admission"]["would_reject_decisions_total"] == 1
    assert snapshot["admission"]["shadow_would_reject_decisions_total"] == 0


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
    assert decision.would_admit
    assert decision.enforced
    assert decision.score is None
    assert decision.reason == "no_deadline"
    assert controller.pop_ready() is not None


def test_shadow_admission_records_would_reject_and_correlation_without_rejecting() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(effective_k=0, mu=0.0, enforce=False),
        clock=lambda: 10.0,
    )
    decision = controller.enqueue(
        _pending(
            "shadowed",
            request_class="interactive",
            deadline=20.0,
            admission_correlation_id="client-request-7",
        )
    )
    assert decision is not None
    assert decision.admitted
    assert not decision.would_admit
    assert not decision.enforced
    assert controller.pop_ready().pending.request_id == "shadowed"  # type: ignore[union-attr]

    controller.cancel_request("shadowed")
    admission = controller.snapshot()["admission"]
    assert admission["admitted_total"] == 1
    assert admission["rejected_total"] == 0
    assert admission["actual_rejected_total"] == 0
    assert admission["would_reject_decisions_total"] == 1
    assert admission["shadow_would_reject_decisions_total"] == 1
    assert admission["decision_sequence"] == 1
    assert admission["recent_decisions"][-1] == {
        "decision_sequence": 1,
        "request_id": "shadowed",
        "admission_correlation_id": "client-request-7",
        "request_class": "interactive",
        "phase": "arrival",
        "admitted": True,
        "would_admit": False,
        "enforced": False,
        "score": 0.0,
        "gamma": 0.75,
        "reason": "zero_effective_k",
        "effective_k": 0,
        "mu": 0.0,
        "active_count": 0,
        "queue_position": 0,
        "remaining_budget_s": 10.0,
        "required_returns": None,
        "threshold_s": None,
        "threshold_slack_s": None,
        "score_method": "erlang_empirical",
        "threshold_table_digest": None,
    }


def test_shadow_admission_recheck_observes_failure_without_removing_waiter() -> None:
    now = [0.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            policy="fifo",
            global_wip_limit=1,
            admission=_admission_config(enforce=False).admission,
        ),
        clock=lambda: now[0],
    )
    running = controller.enqueue(_pending("running", request_class="interactive", deadline=100.0))
    assert running is not None and running.would_admit
    assert controller.pop_ready() is not None
    waiting = controller.enqueue(_pending("waiting", request_class="interactive", deadline=1.0))
    assert waiting is not None and waiting.would_admit

    now[0] = 2.0
    assert controller.recheck_admission() == []
    snapshot = controller.snapshot()
    assert snapshot["queued_requests"] == 1
    assert snapshot["admission"]["actual_rejected_total"] == 0
    assert snapshot["admission"]["shadow_would_reject_decisions_total"] == 1
    last = snapshot["admission"]["recent_decisions"][-1]
    assert last["phase"] == "recheck"
    assert last["admitted"] is True
    assert last["would_admit"] is False

    controller.cancel_request("running")
    assert controller.pop_ready().pending.request_id == "waiting"  # type: ignore[union-attr]


def test_shadow_admission_positions_reflect_observed_queue() -> None:
    now = [0.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(mu=0.1, gamma=0.8, enforce=False),
        clock=lambda: now[0],
    )
    stale = controller.enqueue(_pending("stale", request_class="interactive", deadline=1.0))
    assert stale is not None and stale.would_admit

    now[0] = 2.0
    newcomer = controller.enqueue(_pending("new", request_class="interactive", deadline=3.0))
    assert newcomer is not None and not newcomer.would_admit
    assert newcomer.queue_position == 1

    assert controller.recheck_admission() == []
    decisions = controller.snapshot()["admission"]["recent_decisions"][-2:]
    assert [(item["request_id"], item["would_admit"], item["queue_position"]) for item in decisions] == [
        ("stale", False, 0),
        ("new", False, 1),
    ]
    assert controller.snapshot()["queued_requests"] == 2


def test_admission_decision_ring_reports_sequence_gaps_without_growing() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(effective_k=0, mu=0.0, enforce=False),
        clock=lambda: 0.0,
    )
    decision_count = ADMISSION_DECISION_HISTORY_LIMIT + 2
    for index in range(decision_count):
        decision = controller.enqueue(
            _pending(
                f"request-{index}",
                request_class="interactive",
                deadline=10.0,
                admission_correlation_id=f"client-{index}",
            )
        )
        assert decision is not None and decision.admitted and not decision.would_admit

    admission = controller.snapshot()["admission"]
    recent = admission["recent_decisions"]
    assert admission["decision_sequence"] == decision_count
    assert admission["recent_decision_capacity"] == ADMISSION_DECISION_HISTORY_LIMIT
    assert admission["recent_decision_first_sequence"] == 3
    assert admission["recent_decision_last_sequence"] == decision_count
    assert admission["recent_decision_overwritten_total"] == 2
    assert len(recent) == ADMISSION_DECISION_HISTORY_LIMIT
    assert [item["decision_sequence"] for item in recent] == list(range(3, decision_count + 1))


def test_admission_rejects_expired_deadline_and_zero_effective_k() -> None:
    expired = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(),
        clock=lambda: 10.0,
    )
    expired_decision = expired.enqueue(_pending("expired", request_class="interactive", deadline=9.0))
    assert expired_decision is not None
    assert not expired_decision.admitted
    assert not expired_decision.would_admit
    assert expired_decision.enforced
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


def test_threshold_admission_uses_only_lookup_and_records_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _threshold_admission_config(
        effective_k=2,
        mu=1.0,
        service_samples_s=(0.1, 0.2, 0.4),
        gamma=0.75,
        max_required_returns=4,
    )
    table = config.admission.threshold_tables["interactive"]

    def unexpected_score(*args, **kwargs):
        raise AssertionError("threshold mode must not invoke an online scorer")

    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score",
        unexpected_score,
    )
    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score_reference",
        unexpected_score,
    )
    controller = RuntimeQueueController(num_stages=1, config=config, clock=lambda: 0.0)
    threshold = table.thresholds_s[0]
    decision = controller.enqueue(
        _pending(
            "at-threshold",
            request_class="interactive",
            deadline=threshold,
        )
    )
    assert decision is not None and decision.admitted and decision.would_admit
    assert decision.score is None
    assert decision.required_returns == 0
    assert decision.threshold_s == threshold
    assert decision.threshold_slack_s == 0.0

    snapshot = controller.snapshot()["admission"]
    class_snapshot = snapshot["classes"]["interactive"]
    assert snapshot["score_method"] == "erlang_empirical_threshold"
    assert class_snapshot["max_required_returns"] == 4
    assert class_snapshot["threshold_entry_count"] == 5
    assert class_snapshot["threshold_profile_fingerprint"] == table.profile_fingerprint
    assert class_snapshot["threshold_table_digest"] == table.table_digest
    assert snapshot["recent_decisions"][-1]["threshold_slack_s"] == 0.0
    assert snapshot["recent_decisions"][-1]["score_method"] == "erlang_empirical_threshold"
    assert snapshot["recent_decisions"][-1]["threshold_table_digest"] == table.table_digest


def test_threshold_admission_fails_closed_beyond_table_without_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_threshold_admission_config(max_required_returns=2),
        clock=lambda: 0.0,
    )
    monkeypatch.setattr(
        controller,
        "_active_stage_class_count",
        lambda stage_id, request_class: 1,
    )

    def unexpected_score(*args, **kwargs):
        raise AssertionError("overflow must fail closed without an online scorer")

    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score",
        unexpected_score,
    )
    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score_reference",
        unexpected_score,
    )
    decision = controller._evaluate_admission(
        _pending("overflow", request_class="interactive", deadline=100.0),
        queue_position=2,
        phase="arrival",
        now_monotonic_s=0.0,
    )
    assert decision is not None
    assert not decision.admitted
    assert not decision.would_admit
    assert decision.reason == "threshold_table_exhausted"
    assert decision.required_returns == 3
    assert decision.score is None
    assert decision.threshold_s is None
    assert decision.threshold_slack_s is None

    zero_gamma = RuntimeQueueController(
        num_stages=1,
        config=_threshold_admission_config(
            gamma=0.0,
            max_required_returns=0,
        ),
        clock=lambda: 0.0,
    )
    monkeypatch.setattr(
        zero_gamma,
        "_active_stage_class_count",
        lambda stage_id, request_class: 1,
    )
    zero_gamma_decision = zero_gamma._evaluate_admission(
        _pending("zero-gamma-overflow", request_class="interactive", deadline=100.0),
        queue_position=100,
        phase="arrival",
        now_monotonic_s=0.0,
    )
    assert zero_gamma_decision is not None and zero_gamma_decision.admitted
    assert zero_gamma_decision.required_returns == 101
    assert zero_gamma_decision.threshold_s == 0.0


def test_threshold_admission_preserves_special_cases_and_shadow_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_capacity = RuntimeQueueController(
        num_stages=1,
        config=_threshold_admission_config(effective_k=0, mu=0.0),
        clock=lambda: 1.0,
    )
    stopped = no_capacity.enqueue(_pending("stopped", request_class="interactive", deadline=2.0))
    assert stopped is not None and not stopped.admitted
    assert stopped.reason == "zero_effective_k"

    controller = RuntimeQueueController(
        num_stages=1,
        config=_threshold_admission_config(
            service_samples_s=(0.5,),
            enforce=False,
        ),
        clock=lambda: 0.0,
    )
    no_deadline = controller.enqueue(_pending("no-deadline-threshold", request_class="interactive"))
    assert no_deadline is not None and no_deadline.would_admit
    assert no_deadline.reason == "no_deadline"

    monkeypatch.setattr(
        controller,
        "_active_stage_class_count",
        lambda stage_id, request_class: 2,
    )
    above_k = controller._evaluate_admission(
        _pending("above-k", request_class="interactive", deadline=2.0),
        queue_position=0,
        phase="arrival",
        now_monotonic_s=1.0,
    )
    assert above_k is not None and above_k.admitted and not above_k.would_admit
    assert above_k.reason == "active_above_effective_k"

    monkeypatch.setattr(
        controller,
        "_active_stage_class_count",
        lambda stage_id, request_class: 0,
    )
    table = controller.config.admission.threshold_tables["interactive"]
    below = controller._evaluate_admission(
        _pending(
            "shadow-below",
            request_class="interactive",
            deadline=math.nextafter(table.thresholds_s[0], -math.inf),
        ),
        queue_position=0,
        phase="arrival",
        now_monotonic_s=0.0,
    )
    assert below is not None and below.admitted and not below.would_admit
    assert below.reason == "score_below_gamma"


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


def test_stage_zero_limit_does_not_dispatch_update_before_initial() -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            policy="edf",
            stage_class_wip_limits={0: {"interactive": 1}},
            admission=_admission_config(gamma=0.1).admission,
        ),
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


def test_admission_uses_shared_stage_occupancy_not_end_to_end_requests() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=_admission_config(
            effective_k=1,
            mu=1.0,
            service_samples_s=(0.0,),
            gamma=0.75,
        ),
        clock=lambda: 0.0,
    )

    first = controller.enqueue(_pending("first", request_class="interactive", deadline=100.0))
    assert first is not None and first.admitted
    assert controller.pop_ready() is not None
    controller.acquire_immediate(
        _pending(
            "first",
            stage_id=1,
            starts_request=False,
            request_class="interactive",
            required_active_stage_id=0,
        )
    )
    assert controller.release_stage("first", 0)
    assert controller.snapshot()["active_requests"] == 1
    assert controller.snapshot()["active_by_stage_class"] == {"1": {"interactive": 1}}

    decision = controller.enqueue(_pending("second", request_class="interactive", deadline=0.1))
    assert decision is not None
    assert decision.admitted
    assert decision.active_count == 0
    assert decision.score == 1.0


def test_admission_does_not_add_end_to_end_cap_beyond_stage_zero_credit() -> None:
    controller = RuntimeQueueController(
        num_stages=3,
        config=QueueControlConfig(
            enabled=True,
            policy="edf",
            stage_class_wip_limits={0: {"interactive": 1}},
            admission=_admission_config(effective_k=1, gamma=0.1).admission,
        ),
        clock=lambda: 0.0,
    )

    controller.enqueue(_pending("first", request_class="interactive"))
    assert controller.pop_ready().pending.request_id == "first"  # type: ignore[union-attr]
    controller.acquire_immediate(
        _pending(
            "first",
            stage_id=1,
            starts_request=False,
            request_class="interactive",
            required_active_stage_id=0,
        )
    )
    assert controller.release_stage("first", 0)

    second = controller.enqueue(_pending("second", request_class="interactive"))
    assert second is not None and second.admitted and second.active_count == 0
    assert controller.pop_ready().pending.request_id == "second"  # type: ignore[union-attr]
    assert controller.snapshot()["active_requests"] == 2

    third = controller.enqueue(_pending("third", request_class="interactive"))
    assert third is not None and third.admitted and third.active_count == 1
    assert controller.pop_ready() is None
    assert controller.snapshot()["blocked_by_limit"] == {"stage_class": 1}

    assert controller.release_stage("second", 0)
    assert controller.pop_ready().pending.request_id == "third"  # type: ignore[union-attr]


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


def test_threshold_arrival_retains_authoritative_edf_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=_threshold_admission_config(mu=0.1, gamma=0.8),
        clock=lambda: now[0],
    )
    stale = controller.enqueue(_pending("stale", request_class="interactive", deadline=1.0))
    assert stale is not None and stale.admitted

    def unexpected_score(*args, **kwargs):
        raise AssertionError("threshold EDF fallback must remain lookup-only")

    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score",
        unexpected_score,
    )
    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score_reference",
        unexpected_score,
    )
    now[0] = 2.0
    newcomer = controller.enqueue(_pending("new", request_class="interactive", deadline=3.0))
    assert newcomer is not None and newcomer.admitted
    assert newcomer.queue_position == 0

    rejected = controller.recheck_admission()
    assert [item.pending.request_id for item in rejected] == ["stale"]


def test_arrival_fast_path_matches_scalar_legacy_decisions_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(18402)

    def run_trace(*, scalar_reference: bool, deadlines: list[float], newcomer_deadline: float):
        now = [2.0]
        controller = RuntimeQueueController(
            num_stages=1,
            config=_admission_config(
                effective_k=4,
                mu=0.7,
                service_samples_s=tuple(0.05 + index * 0.01 for index in range(64)),
                gamma=0.65,
            ),
            clock=lambda: now[0],
        )
        for index, deadline in enumerate(deadlines):
            candidate = _pending(
                f"waiting-{index}",
                request_class="interactive",
                deadline=deadline,
            )
            controller._stamp_pending(candidate)
            controller._pending.append(candidate)

        newcomer = _pending(
            "newcomer",
            request_class="interactive",
            deadline=newcomer_deadline,
        )
        controller._stamp_pending(newcomer)
        if scalar_reference:
            candidates = [*controller._admission_candidates("interactive"), newcomer]
            candidates.sort(key=controller._admission_order_key)
            queue_position = 0
            decision = None
            with monkeypatch.context() as patch:
                patch.setattr(
                    queue_control_module,
                    "erlang_empirical_admission_score",
                    erlang_empirical_admission_score_reference,
                )
                for candidate in candidates:
                    if candidate is newcomer:
                        decision = controller._evaluate_admission(
                            newcomer,
                            queue_position=queue_position,
                            phase="arrival",
                            now_monotonic_s=now[0],
                        )
                        break
                    prior = controller._evaluate_admission(
                        candidate,
                        queue_position=queue_position,
                        phase="recheck",
                        now_monotonic_s=now[0],
                    )
                    if prior is None or prior.admitted:
                        queue_position += 1
        else:
            decision = controller._evaluate_arrival_admission(newcomer)
        assert decision is not None
        if decision.admitted:
            controller._pending.append(newcomer)
        rejected = controller.recheck_admission()
        dispatch_order: list[str] = []
        while (acquired := controller.pop_ready()) is not None:
            dispatch_order.append(acquired.pending.request_id)
        return (
            (decision.admitted, decision.would_admit, decision.reason),
            [(item.pending.request_id, item.decision.reason) for item in rejected],
            dispatch_order,
        )

    for _ in range(100):
        deadlines = [rng.uniform(0.0, 8.0) for _ in range(rng.randint(0, 40))]
        newcomer_deadline = rng.uniform(0.0, 8.0)
        assert run_trace(
            scalar_reference=False,
            deadlines=deadlines,
            newcomer_deadline=newcomer_deadline,
        ) == run_trace(
            scalar_reference=True,
            deadlines=deadlines,
            newcomer_deadline=newcomer_deadline,
        )


@pytest.mark.parametrize("queue_size", [10, 50, 100, 500])
@pytest.mark.parametrize("sample_count", [64, 576, 2000])
def test_successful_arrival_scores_only_the_new_request(
    monkeypatch: pytest.MonkeyPatch,
    queue_size: int,
    sample_count: int,
) -> None:
    controller = RuntimeQueueController(
        num_stages=1,
        config=_admission_config(
            effective_k=8,
            mu=2.0,
            service_samples_s=tuple(0.1 + index / sample_count for index in range(sample_count)),
            gamma=0.1,
        ),
        clock=lambda: 0.0,
    )
    for index in range(queue_size):
        candidate = _pending(
            f"waiting-{index}",
            request_class="interactive",
            deadline=10_000.0 + index,
        )
        controller._stamp_pending(candidate)
        controller._pending.append(candidate)

    score_calls = 0
    original = queue_control_module.erlang_empirical_admission_score

    def counted_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        queue_control_module,
        "erlang_empirical_admission_score",
        counted_score,
    )
    decision = controller.enqueue(
        _pending(
            "newcomer",
            request_class="interactive",
            deadline=20_000.0,
        )
    )
    assert decision is not None and decision.admitted
    assert score_calls == 1


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


def test_stage_class_runtime_telemetry_records_live_and_completed_time() -> None:
    now = [1.0]
    controller = RuntimeQueueController(
        num_stages=1,
        config=QueueControlConfig(
            enabled=True,
            stage_class_wip_limits={0: {"text": 1}},
        ),
        clock=lambda: now[0],
    )

    controller.enqueue(
        _pending(
            "request",
            request_class="text",
            admission_correlation_id="client-record-7",
        )
    )
    assert controller.pop_ready() is not None

    now[0] = 2.5
    live = controller.snapshot()
    assert live["active_time_s_by_stage_class_total"] == {"0": {"text": pytest.approx(1.5)}}
    assert live["completed_by_stage_class_total"] == {"0": {"text": 0}}
    assert live["service_time_s_by_stage_class_total"] == {"0": {"text": 0}}
    assert live["admitted_arrivals_by_class_total"] == {"text": 1}

    now[0] = 4.0
    assert controller.release_stage("request", 0)
    completed = controller.snapshot()
    assert completed["completed_by_stage_class_total"] == {"0": {"text": 1}}
    assert completed["service_time_s_by_stage_class_total"] == {"0": {"text": pytest.approx(3.0)}}
    assert completed["service_time_s_sum_sq_by_stage_class_total"] == {"0": {"text": pytest.approx(9.0)}}
    assert completed["active_time_s_by_stage_class_total"] == {"0": {"text": pytest.approx(3.0)}}
    assert completed["rollbacks_by_stage_class_total"] == {"0": {"text": 0}}
    assert completed["cancelled_active_by_stage_class_total"] == {"0": {"text": 0}}
    assert completed["stage_class_runtime"] == {
        "schema_version": 1,
        "observation_monotonic_s": 4.0,
        "stages": {
            "0": {
                "text": {
                    "completed_total": 1,
                    "service_time_s_total": pytest.approx(3.0),
                    "service_time_s_sum_sq_total": pytest.approx(9.0),
                    "active_time_s_total": pytest.approx(3.0),
                    "rollback_total": 0,
                    "cancelled_active_total": 0,
                    "cancelled_pending_total": 0,
                }
            }
        },
    }
    assert completed["recent_stage_completion_overwritten_total"] == 0
    assert completed["recent_stage_completion_schema_version"] == 1
    assert completed["recent_stage_completion_first_sequence"] == 1
    assert completed["recent_stage_completion_last_sequence"] == 1
    assert completed["recent_stage_completions"] == [
        {
            "completion_sequence": 1,
            "request_id": "request",
            "logical_request_id": "request",
            "stage_id": 0,
            "request_class": "text",
            "admission_correlation_id": "client-record-7",
            "enqueued_monotonic_s": 1.0,
            "acquired_monotonic_s": 1.0,
            "released_monotonic_s": 4.0,
            "queue_wait_s": 0.0,
            "service_s": 3.0,
        }
    ]

    now[0] = 8.0
    assert not controller.release_stage("request", 0)
    assert controller.snapshot()["completed_by_stage_class_total"] == {"0": {"text": 1}}


def test_stage_class_runtime_telemetry_does_not_count_rollback_as_completion() -> None:
    now = [10.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    acquired = controller.acquire_immediate(_pending("request", request_class="text"))

    now[0] = 12.0
    controller.rollback(acquired)
    snapshot = controller.snapshot()
    assert snapshot["completed_by_stage_class_total"] == {"0": {"text": 0}}
    assert snapshot["service_time_s_by_stage_class_total"] == {"0": {"text": 0}}
    assert snapshot["active_time_s_by_stage_class_total"] == {"0": {"text": pytest.approx(2.0)}}
    assert snapshot["rollbacks_by_stage_class_total"] == {"0": {"text": 1}}
    assert snapshot["cancelled_active_by_stage_class_total"] == {"0": {"text": 0}}


def test_failed_dispatch_is_rollback_and_late_terminal_is_ignored() -> None:
    now = [10.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    acquired = controller.acquire_immediate(_pending("request", request_class="text"))

    now[0] = 12.0
    assert controller.fail_stage_dispatch("request", 0)
    controller.cancel_request("request")
    controller.rollback(acquired)
    now[0] = 15.0
    assert not controller.release_stage("request", 0)

    snapshot = controller.snapshot()
    assert snapshot["completed_by_stage_class_total"] == {"0": {"text": 0}}
    assert snapshot["service_time_s_by_stage_class_total"] == {"0": {"text": 0}}
    assert snapshot["active_time_s_by_stage_class_total"] == {"0": {"text": pytest.approx(2.0)}}
    assert snapshot["rollbacks_by_stage_class_total"] == {"0": {"text": 1}}
    assert snapshot["cancelled_active_by_stage_class_total"] == {"0": {"text": 0}}


def test_failed_update_to_existing_stage_remains_a_cancellation() -> None:
    now = [10.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    controller.acquire_immediate(_pending("request", request_class="text"))
    update = controller.acquire_immediate(_pending("request", starts_request=False, request_class="text"))
    assert not update.acquired_stage

    now[0] = 12.0
    # Mirrors the orchestrator guard: only newly acquired leases are marked as
    # rollback before request-level failure cleanup.
    if update.pending.acquired_stage_for_dispatch:
        controller.fail_stage_dispatch("request", 0)
    controller.cancel_request("request")
    controller.rollback(update)

    snapshot = controller.snapshot()
    assert snapshot["rollbacks_by_stage_class_total"] == {"0": {"text": 0}}
    assert snapshot["cancelled_active_by_stage_class_total"] == {"0": {"text": 1}}
    assert snapshot["dispatch_failures_total"] == 1


def test_stage_class_runtime_telemetry_cancel_closes_parent_and_child_leases() -> None:
    now = [0.0]
    controller = RuntimeQueueController(num_stages=2, clock=lambda: now[0])
    controller.acquire_immediate(_pending("parent", request_class="speech"))
    controller.acquire_immediate(
        _pending(
            "child",
            logical_request_id="parent",
            stage_id=1,
            starts_request=False,
            request_class="untrusted-child-label",
        )
    )

    now[0] = 3.0
    controller.cancel_request("parent")
    snapshot = controller.snapshot()
    assert snapshot["active_by_stage_class"] == {}
    assert snapshot["completed_by_stage_class_total"] == {
        "0": {"speech": 0},
        "1": {"speech": 0},
    }
    assert snapshot["service_time_s_by_stage_class_total"] == {
        "0": {"speech": 0},
        "1": {"speech": 0},
    }
    assert snapshot["active_time_s_by_stage_class_total"] == {
        "0": {"speech": pytest.approx(3.0)},
        "1": {"speech": pytest.approx(3.0)},
    }
    assert snapshot["cancelled_active_by_stage_class_total"] == {
        "0": {"speech": 1},
        "1": {"speech": 1},
    }


def test_stage_cancellation_ring_identifies_active_and_pending_terminals() -> None:
    now = [1.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    controller.acquire_immediate(
        _pending(
            "active",
            request_class="speech",
            admission_correlation_id="client-active",
        )
    )
    controller.enqueue(
        _pending(
            "pending",
            request_class="text",
            admission_correlation_id="client-pending",
        )
    )

    now[0] = 2.0
    controller.cancel_request("active")
    now[0] = 3.0
    controller.cancel_request("pending")
    snapshot = controller.snapshot()

    assert snapshot["cancelled_active_by_stage_class_total"] == {"0": {"speech": 1, "text": 0}}
    assert snapshot["cancelled_pending_by_stage_class_total"] == {"0": {"speech": 0, "text": 1}}
    assert snapshot["recent_stage_cancellation_schema_version"] == 1
    assert snapshot["recent_stage_cancellation_capacity"] == (RECENT_STAGE_CANCELLATION_HISTORY_LIMIT)
    assert snapshot["recent_stage_cancellation_overwritten_total"] == 0
    assert snapshot["recent_stage_cancellation_first_sequence"] == 1
    assert snapshot["recent_stage_cancellation_last_sequence"] == 2
    assert [
        {
            key: row[key]
            for key in (
                "cancellation_sequence",
                "outcome",
                "request_id",
                "logical_request_id",
                "stage_id",
                "request_class",
                "admission_correlation_id",
            )
        }
        for row in snapshot["recent_stage_cancellations"]
    ] == [
        {
            "cancellation_sequence": 1,
            "outcome": "cancelled_active",
            "request_id": "active",
            "logical_request_id": "active",
            "stage_id": 0,
            "request_class": "speech",
            "admission_correlation_id": "client-active",
        },
        {
            "cancellation_sequence": 2,
            "outcome": "cancelled_pending",
            "request_id": "pending",
            "logical_request_id": "pending",
            "stage_id": 0,
            "request_class": "text",
            "admission_correlation_id": "client-pending",
        },
    ]


def test_stage_cancellation_ring_excludes_downstream_and_companion_leases() -> None:
    controller = RuntimeQueueController(num_stages=2)
    controller.acquire_immediate(_pending("parent", request_class="speech", admission_correlation_id="client"))
    controller.acquire_immediate(
        _pending(
            "child",
            logical_request_id="parent",
            stage_id=1,
            starts_request=False,
            request_class="speech",
            admission_correlation_id="client",
        )
    )
    controller.cancel_request("parent")

    snapshot = controller.snapshot()
    assert snapshot["cancelled_active_by_stage_class_total"] == {
        "0": {"speech": 1},
        "1": {"speech": 1},
    }
    cancellations = snapshot["recent_stage_cancellations"]
    assert len(cancellations) == 1
    assert cancellations[0]["stage_id"] == 0
    assert cancellations[0]["request_id"] == cancellations[0]["logical_request_id"]


def test_stage0_companion_cancellation_does_not_duplicate_identity_ring() -> None:
    controller = RuntimeQueueController(num_stages=1)
    controller.acquire_immediate(_pending("parent", request_class="speech", admission_correlation_id="client"))
    controller.acquire_immediate(
        _pending(
            "companion",
            logical_request_id="parent",
            starts_request=False,
            request_class="speech",
            admission_correlation_id="client",
        )
    )
    controller.cancel_request("parent")

    snapshot = controller.snapshot()
    assert snapshot["cancelled_active_by_stage_class_total"] == {"0": {"speech": 2}}
    assert [row["request_id"] for row in snapshot["recent_stage_cancellations"]] == ["parent"]


def test_repeated_stage_update_does_not_reset_service_timer_or_double_retire() -> None:
    now = [1.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    controller.acquire_immediate(_pending("request", request_class="speech"))

    now[0] = 2.0
    update = controller.acquire_immediate(_pending("request", starts_request=False, request_class="speech"))
    assert not update.acquired_stage
    controller.rollback(update)

    now[0] = 5.0
    assert controller.release_stage("request", 0)
    snapshot = controller.snapshot()
    assert snapshot["completed_by_stage_class_total"] == {"0": {"speech": 1}}
    assert snapshot["service_time_s_by_stage_class_total"] == {"0": {"speech": pytest.approx(4.0)}}
    assert snapshot["active_time_s_by_stage_class_total"] == {"0": {"speech": pytest.approx(4.0)}}
    assert snapshot["rollbacks_by_stage_class_total"] == {"0": {"speech": 0}}


def test_recent_stage_completions_are_bounded_and_count_overwrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(queue_control_module, "RECENT_STAGE_COMPLETION_HISTORY_LIMIT", 2)
    now = [0.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    for index in range(3):
        request_id = f"request-{index}"
        controller.acquire_immediate(
            _pending(
                request_id,
                request_class="text",
                admission_correlation_id=f"correlation-{index}",
            )
        )
        now[0] += 1.0
        assert controller.release_stage(request_id, 0)
        controller.cancel_request(request_id)

    snapshot = controller.snapshot()
    assert snapshot["recent_stage_completion_capacity"] == 2
    assert snapshot["recent_stage_completion_overwritten_total"] == 1
    assert snapshot["recent_stage_completion_first_sequence"] == 2
    assert snapshot["recent_stage_completion_last_sequence"] == 3
    assert [row["request_id"] for row in snapshot["recent_stage_completions"]] == [
        "request-1",
        "request-2",
    ]


def test_default_snapshot_completion_history_has_bounded_serialized_size(tmp_path) -> None:
    now = [0.0]
    controller = RuntimeQueueController(num_stages=1, clock=lambda: now[0])
    for index in range(RECENT_STAGE_COMPLETION_HISTORY_LIMIT + 64):
        request_id = f"request-{index}"
        controller.acquire_immediate(
            _pending(
                request_id,
                request_class="text",
                admission_correlation_id=f"correlation-{index}",
            )
        )
        now[0] += 0.01
        assert controller.release_stage(request_id, 0)
        controller.cancel_request(request_id)

    encoded = json.dumps(controller.snapshot(), separators=(",", ":")).encode()
    output = tmp_path / "snapshot.json"
    output.write_bytes(encoded)
    assert len(encoded) < 256 * 1024
    assert len(controller.snapshot()["recent_stage_completions"]) == RECENT_STAGE_COMPLETION_HISTORY_LIMIT
