"""Find this computer's LAN IPv4 address for the audience display links.

Only the audience display server (all interfaces, display ports) uses this
address. The operator stays on its own bind address; see ``security.py``.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Mapping, Sequence

TUNNEL_PREFIXES = ("utun", "tun", "tap", "ppp", "ipsec", "wg", "tailscale", "zt")
VIRTUAL_PREFIXES = ("bridge", "vmnet", "vboxnet", "docker", "br-", "veth", "virbr")
PRIVATE_NETS = tuple(ipaddress.ip_network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
SHARED_NET = ipaddress.ip_network("100.64.0.0/10")  # carrier-grade NAT and mesh VPNs


def _route_ok(address: str) -> bool:
    try:
        ip = ipaddress.IPv4Address(address)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip in SHARED_NET)


def choose_lan_ipv4(interfaces: Mapping[str, Sequence[str]], route_ip: str | None) -> str | None:
    """Pick the default-route address, else the first private address; skip tunnels and link-local."""
    names = [name for name in interfaces if not name.lower().startswith(TUNNEL_PREFIXES)]
    if route_ip and _route_ok(route_ip) and any(route_ip in interfaces[name] for name in names):
        return route_ip
    # Real adapters before virtual bridges, then by name, so the choice is stable.
    for name in sorted(names, key=lambda n: (n.lower().startswith(VIRTUAL_PREFIXES), n)):
        for address in interfaces[name]:
            if _route_ok(address) and any(ipaddress.IPv4Address(address) in net for net in PRIVATE_NETS):
                return address
    return None


def _default_route_ipv4() -> str | None:
    # connect() on a UDP socket only asks the kernel for a route; it sends no packet.
    # The target matches dry_run_ab.get_local_ip, so the address matches the one it prints.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
    except OSError:
        return None


def lan_ipv4() -> str | None:
    """Return a LAN IPv4 address of this computer, or ``None`` when there is none."""
    try:
        import psutil

        stats = psutil.net_if_stats()
        interfaces = {
            name: [entry.address for entry in entries if entry.family == socket.AF_INET]
            for name, entries in psutil.net_if_addrs().items()
            if name in stats and stats[name].isup
        }
    except Exception:  # an interface query failure only removes the optional LAN links
        return None
    return choose_lan_ipv4(interfaces, _default_route_ipv4())
