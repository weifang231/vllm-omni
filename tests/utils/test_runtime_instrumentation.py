# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import time

import vllm_omni.utils.runtime_instrumentation as runtime_instrumentation_module
from vllm_omni.utils.runtime_instrumentation import (
    RUNTIME_CONTROL_FILE_ENV,
    RUNTIME_METRICS_DIR_ENV,
    RuntimeInstrumentation,
)


def test_control_reader_keeps_last_valid_document(monkeypatch, tmp_path) -> None:
    control_path = tmp_path / "control.json"
    control_path.write_text('{"queue_control":{"enabled":true}}', encoding="utf-8")
    monkeypatch.setenv(RUNTIME_CONTROL_FILE_ENV, str(control_path))

    instrumentation = RuntimeInstrumentation(
        engine="test",
        component="queue",
        stage_id="pipeline",
    )
    assert instrumentation.read_control()["queue_control"]["enabled"] is True

    control_path.write_text("{", encoding="utf-8")
    assert instrumentation.read_control()["queue_control"]["enabled"] is True


def test_snapshot_is_atomic_and_bounded_cardinality(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(RUNTIME_METRICS_DIR_ENV, str(tmp_path))
    instrumentation = RuntimeInstrumentation(
        engine="test",
        component="queue",
        stage_id="pipeline",
    )

    assert instrumentation.write_snapshot({"active_requests": 2}, force=True)
    snapshot_path = instrumentation.snapshot_path
    assert snapshot_path is not None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["engine"] == "test"
    assert payload["component"] == "queue"
    assert payload["active_requests"] == 2
    assert payload["snapshot_schema_version"] == 1
    assert payload["runtime_id"] == instrumentation.runtime_id
    assert payload["snapshot_sequence"] == 1
    assert payload["monotonic_time_s"] >= 0
    assert not list(tmp_path.glob("*.tmp"))

    assert instrumentation.write_snapshot({"active_requests": 3}, force=True)
    next_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert next_payload["runtime_id"] == payload["runtime_id"]
    assert next_payload["snapshot_sequence"] == 2
    assert next_payload["monotonic_time_s"] >= payload["monotonic_time_s"]


def test_snapshot_payload_cannot_spoof_causal_envelope(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(RUNTIME_METRICS_DIR_ENV, str(tmp_path))
    instrumentation = RuntimeInstrumentation(
        engine="test",
        component="queue",
        stage_id="pipeline",
    )

    assert instrumentation.write_snapshot(
        {
            "runtime_id": "spoofed",
            "snapshot_sequence": 999,
            "monotonic_time_s": -1,
        },
        force=True,
    )
    snapshot_path = instrumentation.snapshot_path
    assert snapshot_path is not None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["runtime_id"] == instrumentation.runtime_id
    assert payload["snapshot_sequence"] == 1
    assert payload["monotonic_time_s"] >= 0


def test_hot_snapshot_write_cost_stays_bounded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(RUNTIME_METRICS_DIR_ENV, str(tmp_path))
    # Isolate JSON/copy/file-replacement overhead from filesystem durability,
    # whose latency is host-specific and is tested by the normal write path.
    monkeypatch.setattr(runtime_instrumentation_module.os, "fsync", lambda _fd: None)
    instrumentation = RuntimeInstrumentation(
        engine="test",
        component="queue",
        stage_id="pipeline",
    )
    completions = [
        {
            "completion_sequence": index + 1,
            "request_id": f"request-{index}",
            "logical_request_id": f"request-{index}",
            "stage_id": 0,
            "request_class": "text",
            "admission_correlation_id": f"correlation-{index}",
            "enqueued_monotonic_s": float(index),
            "acquired_monotonic_s": float(index) + 0.01,
            "released_monotonic_s": float(index) + 0.02,
            "queue_wait_s": 0.01,
            "service_s": 0.01,
        }
        for index in range(512)
    ]
    payload = {
        "stage_class_runtime": {
            "schema_version": 1,
            "observation_monotonic_s": 1.0,
            "stages": {
                "0": {
                    "text": {
                        "completed_total": 10_000,
                        "service_time_s_total": 100.0,
                        "service_time_s_sum_sq_total": 2.0,
                        "active_time_s_total": 400.0,
                    }
                }
            },
        },
        "recent_stage_completion_capacity": 512,
        "recent_stage_completions": completions,
    }

    started = time.perf_counter()
    for _ in range(10):
        assert instrumentation.write_snapshot(payload, force=True)
    elapsed_s = time.perf_counter() - started
    snapshot_path = instrumentation.snapshot_path
    assert snapshot_path is not None
    assert snapshot_path.stat().st_size < 256 * 1024
    assert elapsed_s < 0.5
