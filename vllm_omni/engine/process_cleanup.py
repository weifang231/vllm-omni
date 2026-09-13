# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Record engine process exits after the process manager finishes shutdown."""

import json
import os
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

from vllm import envs
from vllm.v1.utils import shutdown


def shutdown_stage_processes(
    processes: list[BaseProcess],
    stage_id: int,
    metrics_directory: str | None,
    timeout: float | None = None,
) -> None:
    started = time.monotonic()
    if timeout is None:
        # The parent must outlive worker grace (5s by default), SIGTERM
        # escalation (4s), and EngineCore's own resource teardown.
        timeout = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS + 4.0 + 5.0
    try:
        shutdown(processes, timeout=timeout)
    finally:
        # Reap an escalated child before recording its actual exit status.
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5.0)
        if metrics_directory:
            directory = Path(metrics_directory) / "process_cleanup"
            directory.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": 1,
                "owner_pid": os.getpid(),
                "stage_id": stage_id,
                "elapsed_s": time.monotonic() - started,
                "processes": [
                    {"pid": process.pid, "name": process.name, "returncode": process.exitcode}
                    for process in processes
                ],
            }
            pids = "-".join(str(process.pid) for process in processes)
            path = directory / f"{os.getpid()}-stage{stage_id}-{pids}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(temporary, path)
