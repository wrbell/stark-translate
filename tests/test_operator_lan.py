"""LAN audience links for an operator page opened on loopback; no real interface is read."""

from __future__ import annotations

import pytest

from operator_app.lan import choose_lan_ipv4

WIFI = {"lo0": ["127.0.0.1"], "en0": ["192.168.1.20"], "utun3": ["10.8.0.5"], "en5": ["169.254.10.2"]}


@pytest.mark.parametrize(
    ("interfaces", "route", "expected"),
    [
        (WIFI, "192.168.1.20", "192.168.1.20"),  # the default-route address wins
        ({**WIFI, "en1": ["10.0.0.7"]}, "10.0.0.7", "10.0.0.7"),
        ({"en0": ["198.51.100.9"]}, "198.51.100.9", "198.51.100.9"),  # a public LAN address on the default route
        (WIFI, "10.8.0.5", "192.168.1.20"),  # full-tunnel VPN route: fall back to the LAN address
        ({"lo0": ["127.0.0.1"], "en0": ["100.72.1.2"]}, "100.72.1.2", None),  # shared (CGNAT) space
        (WIFI, None, "192.168.1.20"),  # offline route lookup: first private address
        ({"bridge100": ["192.168.64.1"], "en0": ["172.16.4.9"]}, None, "172.16.4.9"),
        ({"lo0": ["127.0.0.1"], "en5": ["169.254.10.2"], "utun3": ["10.8.0.5"]}, None, None),
        ({}, "127.0.0.1", None),
        (WIFI, "192.0.2.50", "192.168.1.20"),  # a route address on no listed interface is ignored
    ],
)
def test_choose_lan_ipv4(interfaces, route, expected):
    assert choose_lan_ipv4(interfaces, route) == expected


@pytest.mark.parametrize("lan", ["192.168.1.20", None])
def test_capabilities_add_lan_audience_urls_only_when_found(monkeypatch, lan):
    from fastapi.testclient import TestClient

    import operator_app.main as main
    from settings import settings

    monkeypatch.setattr(settings.server, "http_port", 18080)
    monkeypatch.setattr(settings.server, "ws_port", 18765)
    monkeypatch.setattr(main, "lan_ipv4", lambda: lan)
    caps = TestClient(main.app).get("/api/capabilities").json()
    # The loopback links stay as they are for older pages and the projector browser.
    assert caps["audience_urls"]["mobile"] == "http://localhost:18080/displays/mobile_display.html?port=18765"
    if lan is None:
        assert "audience_lan_urls" not in caps and "audience_lan_host" not in caps
        return
    assert caps["audience_lan_host"] == lan
    lan_urls = caps["audience_lan_urls"]
    assert set(lan_urls) == set(caps["audience_urls"])
    assert lan_urls["mobile"] == "http://192.168.1.20:18080/displays/mobile_display.html?port=18765"
    assert all(":9000" not in url and "/operator" not in url for url in lan_urls.values())
