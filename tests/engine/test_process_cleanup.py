import json
import multiprocessing
import os
import signal
from unittest.mock import patch

from vllm_omni.engine.process_cleanup import shutdown_stage_processes


def _ignore_termination(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        signal.pause()


def test_receipt_records_internal_sigkill(tmp_path):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=_ignore_termination, args=(ready,))
    process.start()
    try:
        assert ready.wait(5)
        shutdown_stage_processes([process], 2, str(tmp_path), timeout=0.05)
        record = json.loads(next((tmp_path / "process_cleanup").glob("*.json")).read_text())
        assert record["owner_pid"] == os.getpid()
        assert record["processes"][0]["pid"] == process.pid
        assert record["processes"][0]["returncode"] == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)


def test_disabled_metrics_preserves_shutdown_arguments():
    with patch("vllm_omni.engine.process_cleanup.shutdown") as stop:
        shutdown_stage_processes([], 0, None, timeout=7.5)
    stop.assert_called_once_with([], timeout=7.5)
