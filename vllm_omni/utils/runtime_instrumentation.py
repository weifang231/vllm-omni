# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in file-based controls and runtime snapshots for experiments.

The helpers are inert unless a ``VLLM_OMNI_RUNTIME_*`` environment variable is
set. Control writers should atomically replace the JSON file; readers retain
the most recent valid object across a transient partial or malformed write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_CONTROL_FILE_ENV = "VLLM_OMNI_RUNTIME_CONTROL_FILE"
RUNTIME_CONTROL_INTERVAL_ENV = "VLLM_OMNI_RUNTIME_CONTROL_INTERVAL_S"
RUNTIME_METRICS_DIR_ENV = "VLLM_OMNI_RUNTIME_METRICS_DIR"
RUNTIME_METRICS_INTERVAL_ENV = "VLLM_OMNI_RUNTIME_METRICS_INTERVAL_S"


def _nonnegative_interval_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default
    if value < 0:
        logger.warning("Ignoring negative %s=%r", name, raw)
        return default
    return value


class RuntimeInstrumentation:
    """Read a shared JSON control file and emit a per-process JSON snapshot."""

    def __init__(self, *, engine: str, component: str, stage_id: int | str):
        self.engine = str(engine)
        self.component = str(component)
        self.stage_id = stage_id
        metrics_dir = os.environ.get(RUNTIME_METRICS_DIR_ENV, "").strip()
        control_file = os.environ.get(RUNTIME_CONTROL_FILE_ENV, "").strip()
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        self.control_file = Path(control_file) if control_file else None
        self.control_interval_s = _nonnegative_interval_from_env(
            RUNTIME_CONTROL_INTERVAL_ENV,
            0.1,
        )
        self.snapshot_interval_s = _nonnegative_interval_from_env(
            RUNTIME_METRICS_INTERVAL_ENV,
            1.0,
        )
        # ``runtime_id`` distinguishes a restarted process even if the OS
        # happens to reuse its PID.  The sequence and monotonic timestamp form
        # the causal clock consumed by the external allocator; wall time is
        # retained only for correlating the snapshot with experiment logs.
        self.runtime_id = uuid.uuid4().hex
        self._snapshot_sequence = 0
        self._last_snapshot_monotonic = float("-inf")
        self._snapshot_lock = threading.Lock()
        self._control_signature: tuple[int, int, int] | None = None
        self._control: dict[str, Any] = {}
        self._warned: set[str] = set()

    @property
    def control_enabled(self) -> bool:
        return self.control_file is not None

    @property
    def metrics_enabled(self) -> bool:
        return self.metrics_dir is not None

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def read_control(self) -> dict[str, Any]:
        """Return the latest valid control object without exposing partial writes."""
        if self.control_file is None:
            return {}
        try:
            stat = self.control_file.stat()
        except FileNotFoundError:
            self._control_signature = None
            self._control = {}
            return {}
        except OSError as exc:
            self._warn_once("control-stat", "Cannot stat runtime control file %s: %s", self.control_file, exc)
            return self._control

        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature == self._control_signature:
            return self._control
        try:
            with self.control_file.open(encoding="utf-8") as control_stream:
                candidate = json.load(control_stream)
            if not isinstance(candidate, dict):
                raise TypeError("top-level JSON value must be an object")
        except (OSError, TypeError, ValueError) as exc:
            self._control_signature = signature
            self._warn_once(
                f"control-read:{signature}",
                "Ignoring invalid runtime control file %s: %s",
                self.control_file,
                exc,
            )
            return self._control

        self._control_signature = signature
        self._control = candidate
        return self._control

    @property
    def snapshot_path(self) -> Path | None:
        if self.metrics_dir is None:
            return None
        filename = f"{self.engine}.{self.component}.stage-{self.stage_id}.pid-{os.getpid()}.json"
        return self.metrics_dir / filename

    def snapshot_due(self) -> bool:
        return self.metrics_enabled and time.monotonic() - self._last_snapshot_monotonic >= self.snapshot_interval_s

    def write_snapshot(self, payload: Mapping[str, Any], *, force: bool = False) -> bool:
        """Atomically replace this component's snapshot when its interval expires."""
        output_path = self.snapshot_path
        if output_path is None:
            return False
        with self._snapshot_lock:
            now_monotonic = time.monotonic()
            if not force and now_monotonic - self._last_snapshot_monotonic < self.snapshot_interval_s:
                return False

            next_sequence = self._snapshot_sequence + 1
            snapshot = dict(payload)
            # Write the authoritative envelope last so a caller cannot spoof
            # the causal clock or process identity through its payload.
            snapshot.update(
                {
                    "snapshot_schema_version": 1,
                    "timestamp_s": time.time(),
                    "monotonic_time_s": now_monotonic,
                    "runtime_id": self.runtime_id,
                    "snapshot_sequence": next_sequence,
                    "engine": self.engine,
                    "component": self.component,
                    "stage_id": self.stage_id,
                    "pid": os.getpid(),
                }
            )

            temporary_path: str | None = None
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=output_path.parent,
                    prefix=f".{output_path.name}.{threading.get_ident()}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = temporary.name
                    json.dump(snapshot, temporary, sort_keys=True, separators=(",", ":"))
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, output_path)
                temporary_path = None
                self._snapshot_sequence = next_sequence
                self._last_snapshot_monotonic = now_monotonic
                return True
            except OSError as exc:
                self._warn_once("snapshot-write", "Cannot write runtime snapshot %s: %s", output_path, exc)
                return False
            finally:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
