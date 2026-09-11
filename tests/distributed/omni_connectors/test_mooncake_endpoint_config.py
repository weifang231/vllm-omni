import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import zmq
from vllm.utils.network_utils import make_zmq_path

from vllm_omni.distributed.omni_connectors.connectors import mooncake_transfer_engine_connector as module
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.distributed.omni_connectors.utils.config import (
    ConnectorSpec,
    OmniTransferConfig,
    stage_receives_chunks,
    stage_sends_async_output,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import chunk_zmq_port
from vllm_omni.engine.stage_init_utils import get_stage_connector_spec


def test_async_pipeline_assigns_both_directions_to_middle_stage():
    spec = ConnectorSpec("MooncakeTransferEngineConnector", {"zmq_port": 50051})
    transfer = OmniTransferConfig(connectors={("0", "1"): spec, ("1", "2"): spec})
    roles = [(0, False, True, "sender"), (1, True, True, "both"), (2, True, False, "receiver")]
    for stage_id, receives, sends, role in roles:
        config = get_stage_connector_spec(transfer, stage_id, async_chunk=True)
        assert config["extra"]["role"] == role
        model = SimpleNamespace(stage_id=stage_id, stage_connector_config=config)
        assert stage_receives_chunks(model) == receives
        assert stage_sends_async_output(model) == sends
    assert spec.extra == {"zmq_port": 50051}


def test_mismatched_async_edges_are_rejected():
    transfer = OmniTransferConfig(connectors={
        ("0", "1"): ConnectorSpec("MooncakeTransferEngineConnector", {"zmq_port": 50051}),
        ("1", "2"): ConnectorSpec("MooncakeTransferEngineConnector", {"zmq_port": 51051}),
    })
    with pytest.raises(ValueError, match="shared connector specification"):
        get_stage_connector_spec(transfer, 1, async_chunk=True)


def test_replicas_bind_distinct_chunk_sockets(monkeypatch):
    context = zmq.Context()
    sockets = []
    configs = []
    # Reserve an available port block before exercising the allocator. Shared
    # test hosts may already have listeners on the default serving ports.
    reserved = {}
    for base in range(20000, 30000, 37):
        try:
            for replica in range(4):
                for stage in (0, 1):
                    port = base + replica * 1024 + stage
                    listener = context.socket(zmq.ROUTER)
                    sockets.append(listener)
                    listener.bind(f"tcp://127.0.0.1:{port}")
                    reserved[port] = listener
            break
        except zmq.ZMQError as error:
            if error.errno != zmq.EADDRINUSE:
                raise
            for listener in sockets:
                listener.close(linger=0)
            sockets.clear()
            reserved.clear()
    assert len(reserved) == 8

    def bind(spec):
        configs.append(spec.extra)
        return reserved.pop(spec.extra["zmq_port"])
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter"
        ".OmniConnectorFactory.create_connector", bind,
    )
    try:
        for replica in range(4):
            monkeypatch.setenv("VLLM_OMNI_REPLICA_ID", str(replica))
            for stage in (0, 1):
                original = {"name": "MooncakeTransferEngineConnector", "extra": {"zmq_port": base}}
                model = SimpleNamespace(stage_id=stage, stage_connector_config=original)
                OmniChunkTransferAdapter.create_connector(model)
                assert configs[-1]["zmq_port"] == chunk_zmq_port(base, stage, replica)
                assert original["extra"]["zmq_port"] == base
        assert len({c["zmq_port"] for c in configs}) == 8
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()


@pytest.mark.parametrize("stage,replica,base", [(16, 0, 50051), (0, -1, 50051), (0, 16, 50051)])
def test_invalid_chunk_endpoints_fail_before_startup(stage, replica, base):
    with pytest.raises(ValueError):
        chunk_zmq_port(base, stage, replica)


def test_explicit_worker_host_overrides_auto_detection():
    cls = module.MooncakeTransferEngineConnector
    with patch.object(module, "TransferEngine", object()), \
         patch.object(cls, "_init_engine"), patch.object(cls, "_init_pool"), \
         patch.object(cls, "_init_listener"), \
         patch.object(cls, "_get_local_ip", side_effect=AssertionError("Unexpected auto detection")), \
         patch.dict(os.environ, OMNI_STAGE_HOST="worker.example.com"):
        connector = cls({"host": "auto", "host_env": "OMNI_STAGE_HOST"})
        connector._closed = True
        assert connector.host == "worker.example.com"
        with pytest.raises(ValueError, match="host or host_env"):
            cls({"host": "127.0.0.1", "host_env": "OMNI_STAGE_HOST"})


def test_ipv6_context_exchanges_control_messages():
    context = zmq.Context()
    context.set(zmq.IPV6, 1)
    try:
        with context.socket(zmq.REP) as server, context.socket(zmq.REQ) as client:
            for socket in (server, client):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 2000)
                socket.setsockopt(zmq.SNDTIMEO, 2000)
            port = server.bind_to_random_port("tcp://[::1]")
            client.connect(make_zmq_path("tcp", "::1", port))
            client.send(b"request")
            assert server.recv() == b"request"
            server.send(b"ack")
            assert client.recv() == b"ack"
    finally:
        context.term()


@pytest.mark.skipif(module.TransferEngine is None, reason="Mooncake TransferEngine is not installed")
def test_cpu_tcp_chunks_cross_selected_senders_and_middle_stage(monkeypatch):
    import torch
    from tests.helpers.runtime import get_open_port
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

    # TCP can transfer ordinary CPU memory; this test needs no CUDA context.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    connectors = []
    try:
        for role in ("sender", "sender", "both", "receiver"):
            connectors.append(module.MooncakeTransferEngineConnector({
                "role": role, "host": "127.0.0.1", "zmq_port": get_open_port(),
                "protocol": "tcp", "memory_pool_size": 1024 * 1024, "memory_pool_device": "cpu",
            }))
        first, second, middle, final = connectors
        for value, sender in [(7, second), (11, first), (13, second)]:
            payload = OmniPayloadStruct(codes=CodesStruct(audio=torch.tensor([[value]])),
                                        meta=MetaStruct(finished=torch.tensor(True)))
            key = f"request-{value}_0_0"
            assert sender.put("0", "1", key, payload)[0]
            wrong = first if sender is second else second
            assert middle.get("0", "1", key, {"source_host": wrong.host, "source_port": wrong.zmq_port}) is None
            received, _ = middle.get("0", "1", key, {"source_host": sender.host, "source_port": sender.zmq_port})
            assert torch.equal(received["codes"]["audio"], payload.codes.audio)
            assert middle.put("1", "2", key, received)[0]
            output, _ = final.get("1", "2", key, {"source_host": middle.host, "source_port": middle.zmq_port})
            assert torch.equal(output["codes"]["audio"], payload.codes.audio)
            assert bool(output["meta"]["finished"])
        assert not first._local_buffers and not second._local_buffers and not middle._local_buffers
    finally:
        for connector in reversed(connectors):
            connector.close()
