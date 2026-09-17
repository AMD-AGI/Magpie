import sys

from Magpie.mcp import __main__ as entrypoint
from Magpie.mcp import server


def test_mcp_main_stdio_and_http(monkeypatch):
    calls = []
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["magpie-mcp"])
    entrypoint.main()
    assert calls[-1] == {"transport": "stdio"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "magpie-mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ],
    )
    entrypoint.main()
    assert calls[-1] == {"transport": "streamable-http"}
    assert entrypoint.os.environ["MAGPIE_HOST"] == "127.0.0.1"
    assert entrypoint.os.environ["MAGPIE_PORT"] == "9000"
