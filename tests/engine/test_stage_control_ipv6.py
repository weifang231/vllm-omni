# SPDX-License-Identifier: Apache-2.0
"""Registration and membership must work over IPv6-only stage networks."""

import socket
import time

import pytest

from vllm_omni.engine import stage_engine_startup as startup
from vllm_omni.distributed.omni_coordinator import OmniCoordinator, OmniCoordClientForStage
from vllm_omni.utils.network import resolve_tcp_host

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hostname_prefers_ipv6_over_container_alias(monkeypatch):
    def resolve(host, port, family, kind):
        assert host == "worker.example"
        assert family == socket.AF_INET6
        return [(family, kind, 6, "", ("2001:db8::1", 0, 0, 0))]
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    assert resolve_tcp_host("worker.example") == "2001:db8::1"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_stage_registration_and_advertised_handshake(host, monkeypatch):
    monkeypatch.setattr(startup, "_DEFAULT_STARTUP_TIMEOUT_S", 3)
    monkeypatch.setattr(startup, "_POLL_PERIOD_MS", 20)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    server = startup.OmniMasterServer(host, port, [0])
    server.start()
    try:
        result = startup.register_stage_with_omni_master(
            omni_master_address=host, omni_master_port=port, omni_stage_id=0)
        prefix = "tcp://[::1]:" if host == "::1" else "tcp://127.0.0.1:"
        assert result.handshake_address.startswith(prefix)
        assert result.input_address.startswith(prefix)
        assert result.output_address.startswith(prefix)
        assert server.get_allocation(0).replica_host == host
    finally:
        server.stop()


def test_ipv6_coordinator_receives_stage_membership():
    coordinator = OmniCoordinator("tcp://[::1]:0", "tcp://[::1]:0", heartbeat_timeout=1000)
    client = None
    try:
        client = OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://[::1]:19001", "tcp://[::1]:19002", 0)
        deadline = time.monotonic() + 3
        while not coordinator._replicas and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(coordinator._replicas) == 1
    finally:
        if client is not None:
            client.close()
        coordinator.close()
