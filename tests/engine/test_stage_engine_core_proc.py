from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

from vllm_omni.engine import stage_engine_core_proc as stage_module
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


@pytest.mark.parametrize("has_work,exit_code,freezes", [(False, 143, True), (True, 143, False), (False, 1, False)])
def test_only_drained_signal_exit_freezes_after_cleanup(monkeypatch, has_work, exit_code, freezes):
    engine = MagicMock()
    engine.shutdown_state = EngineShutdownState.REQUESTED
    engine.has_work.return_value = has_work
    engine.run_busy_loop.side_effect = SystemExit(exit_code)
    order = []
    engine.shutdown.side_effect = lambda: order.append("shutdown")
    for name in (
        "set_death_signal", "set_process_title", "decorate_logs",
        "maybe_register_config_serialize_by_value", "maybe_apply_cfg_scheduler_patches",
    ):
        monkeypatch.setattr(stage_module, name, MagicMock())
    monkeypatch.setattr(stage_module.signal, "signal", MagicMock())
    monkeypatch.setattr(stage_module, "SignalCallback", MagicMock())
    monkeypatch.setattr(stage_module, "StageEngineCoreProc", MagicMock(return_value=engine))
    monkeypatch.setattr(stage_module.gc, "freeze", lambda: order.append("freeze"))
    with pytest.raises(SystemExit) as error:
        StageEngineCoreProc.run_stage_core()
    assert error.value.code == exit_code
    assert order == (["shutdown", "freeze"] if freezes else ["shutdown"])
