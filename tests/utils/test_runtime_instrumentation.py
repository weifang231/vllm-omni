# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json

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
