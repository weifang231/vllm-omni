import os
from unittest.mock import patch

import pytest
import zmq
from vllm.utils.network_utils import make_zmq_path

from vllm_omni.distributed.omni_connectors.connectors import mooncake_transfer_engine_connector as module


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
