from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from serving import server


def test_main_loads_repo_dotenv_before_refreshing_runtime_and_preserves_process_env(monkeypatch):
    events = []
    old_runtime = server.RUNTIME
    monkeypatch.setattr(server, "RUNTIME", old_runtime)
    monkeypatch.setenv("QWEN_MODEL_PATH", "explicit-process-model-path")
    monkeypatch.delenv("QWEN_ADAPTER_PATH", raising=False)

    def fake_load_dotenv(*, dotenv_path, override, interpolate):
        events.append(("dotenv", Path(dotenv_path), override, interpolate))
        # Mimic python-dotenv with override=False: only fill missing environment values.
        os.environ.setdefault("QWEN_MODEL_PATH", "repo-dotenv-model-path")
        os.environ.setdefault("QWEN_ADAPTER_PATH", "repo-dotenv-adapter-path")

    refreshed_runtime = object()

    def fake_runtime():
        events.append(("runtime", os.environ.get("QWEN_MODEL_PATH"), os.environ.get("QWEN_ADAPTER_PATH")))
        return refreshed_runtime

    def fake_serve():
        events.append(("serve", server.RUNTIME))

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(server, "Runtime", fake_runtime)
    monkeypatch.setattr(server, "serve", fake_serve)

    server.main()

    assert events[0] == ("dotenv", Path(server.__file__).resolve().parents[1] / ".env", False, False)
    assert events[1] == ("runtime", "explicit-process-model-path", "repo-dotenv-adapter-path")
    assert events[2] == ("serve", refreshed_runtime)


def test_native_serve_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("QWEN_BIND_HOST", raising=False)
    monkeypatch.delenv("QWEN_BIND_PORT", raising=False)
    fake_http_server = MagicMock()
    thread = MagicMock()
    monkeypatch.setattr(server, "ThreadingHTTPServer", MagicMock(return_value=fake_http_server))
    monkeypatch.setattr(server.threading, "Thread", MagicMock(return_value=thread))

    server.serve()

    server.ThreadingHTTPServer.assert_called_once_with(("127.0.0.1", 18117), server.Handler)
    thread.start.assert_called_once_with()
    fake_http_server.serve_forever.assert_called_once_with()
