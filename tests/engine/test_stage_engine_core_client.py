# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


def test_chunk_source_uses_remote_replica_host_and_replica_port():
    client = _make_client()
    client.stage_id = 1
    client.replica_id = 3
    client.client_addresses = {"input_address": "tcp://head:10000", "replica_host": "worker-b"}
    client._stage_connector_config = {"name": "MooncakeTransferEngineConnector", "extra": {
        "zmq_port": 50051, "host_env": "OMNI_STAGE_HOST", "role": "both"}}
    source = client.get_chunk_transfer_source()
    assert (source.stage_id, source.replica_id, source.host, source.port) == (1, 3, "worker-b", 53124)
