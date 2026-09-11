# SPDX-License-Identifier: Apache-2.0
"""TCP address normalization for distributed stage control."""

import ipaddress
import socket

from vllm.utils.network_utils import make_zmq_path


def resolve_tcp_host(host: str) -> str:
    """Prefer routable IPv6 over a container-local IPv4 hostname alias."""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        addresses = socket.getaddrinfo(host, None, socket.AF_INET6, socket.SOCK_STREAM)
    except socket.gaierror as error:
        if error.errno not in (socket.EAI_NONAME, socket.EAI_ADDRFAMILY):
            raise
        addresses = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    return addresses[0][4][0]


def tcp_endpoint(host: str, port: int) -> str:
    return make_zmq_path("tcp", resolve_tcp_host(host), port)
