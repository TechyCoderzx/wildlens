from pathlib import Path
import logging

from fastapi.testclient import TestClient
import httpx

from app.main import app
from app.routes import voice


client = TestClient(app)


def test_voice_settings_load_backend_env_file_independent_of_working_directory():
    env_file = Path(voice.settings.model_config["env_file"])

    assert env_file.is_absolute()
    assert env_file == Path(__file__).resolve().parents[1] / ".env"


def test_startup_logs_only_configuration_presence_booleans(caplog):
    caplog.set_level(logging.INFO)
    with TestClient(app):
        pass

    assert "elevenlabs_key_loaded=True" in caplog.text
    assert "voice_id_loaded=True" in caplog.text
    assert "agent_id_loaded=True" in caplog.text
    assert voice.settings.api_key not in caplog.text
    assert voice.settings.agent_id not in caplog.text
    assert voice.settings.voice_id not in caplog.text


class FakeAsyncClient:
    def __init__(self, *, response, **kwargs):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.response

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.response


class RaisingAsyncClient:
    def __init__(self, *, error, **kwargs):
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **kwargs):
        raise self.error

    async def post(self, url, **kwargs):
        raise self.error


def configured(monkeypatch):
    monkeypatch.setattr(voice.settings, "api_key", "test-secret")
    monkeypatch.setattr(voice.settings, "agent_id", "agent-test")
    monkeypatch.setattr(voice.settings, "voice_id", "voice-test")


def test_voice_session_returns_server_issued_token_and_grounded_context(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(200, json={"token": "short-lived-token"})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    result = {
        "status": "uncertain",
        "common_name": "Possible house sparrow",
        "confidence": "low",
        "summary": "Brown back streaking is visible, but the tail is obscured.",
        "evidence_details": [{"feature": "Back streaking", "inference": "Consistent with a sparrow", "support": "weak", "source": "original_image"}],
        "observe_next": "Check the tail from a distance.",
    }
    reply = client.post("/api/v1/voice/session", json={"identification": result})

    assert reply.status_code == 200
    body = reply.json()
    assert body["conversation_token"] == "short-lived-token"
    assert "Possible house sparrow" in body["agent_prompt"]
    assert "user-reported observations" in body["agent_prompt"]
    assert "xi-api-key" not in body["agent_prompt"]


def test_voice_session_requires_credentials_without_calling_provider(monkeypatch):
    monkeypatch.setattr(voice.settings, "api_key", "")
    monkeypatch.setattr(voice.settings, "agent_id", "")

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 503
    assert reply.json()["error"] == {
        "code": "voice_not_configured",
        "stage": "config",
        "message": "Voice isn't configured on the server.",
    }


def test_voice_session_identifies_missing_agent_id(monkeypatch, caplog):
    monkeypatch.setattr(voice.settings, "api_key", "test-secret")
    monkeypatch.setattr(voice.settings, "agent_id", "")
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be called")))

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "voice_not_configured"
    assert reply.json()["error"]["stage"] == "config"
    assert "test-secret" not in caplog.text


def test_voice_session_distinguishes_bad_key_from_agent_access(monkeypatch):
    configured(monkeypatch)
    for status_code, provider_status, expected in (
        (401, "invalid_api_key", "voice_auth_failed"),
        (403, "missing_permissions", "voice_permission_missing"),
    ):
        response = httpx.Response(status_code, json={"detail": {"status": provider_status, "message": "private provider detail"}})
        monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

        reply = client.post("/api/v1/voice/session", json={})

        assert reply.status_code == status_code
        assert reply.json()["error"]["code"] == expected
        assert reply.json()["error"]["stage"] == "agent_token"
        assert "private provider detail" not in reply.text


def test_voice_session_sanitizes_provider_rate_limit(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(429, json={"detail": {"status": "quota_exceeded", "message": "private provider detail"}})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 429
    assert reply.json()["error"]["code"] == "voice_rate_limited"
    assert reply.json()["error"]["stage"] == "agent_token"
    assert "private provider detail" not in reply.text


def test_voice_session_reports_missing_agent_configuration(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(404, json={"detail": {"status": "not_found", "message": "private provider detail"}})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 404
    assert reply.json()["error"]["code"] == "voice_not_found"
    assert "private provider detail" not in reply.text


def test_voice_session_rejects_invalid_identification(monkeypatch):
    configured(monkeypatch)

    reply = client.post("/api/v1/voice/session", json={"identification": {"common_name": "", "status": "certain"}})

    assert reply.status_code == 422


def test_speech_returns_audio_and_does_not_cache(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(200, content=b"mp3-audio", headers={"content-type": "audio/mpeg"})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/speech", json={"text": "A likely identification, based on visible evidence."})

    assert reply.status_code == 200
    assert reply.content == b"mp3-audio"
    assert reply.headers["content-type"].startswith("audio/mpeg")
    assert reply.headers["cache-control"] == "no-store"


def test_speech_requires_voice_id(monkeypatch):
    monkeypatch.setattr(voice.settings, "api_key", "test-secret")
    monkeypatch.setattr(voice.settings, "agent_id", "agent-test")
    monkeypatch.setattr(voice.settings, "voice_id", "")

    reply = client.post("/api/v1/voice/speech", json={"text": "Read this field note."})

    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "voice_not_configured"
    assert reply.json()["error"]["stage"] == "config"


def test_speech_rejects_empty_and_oversized_text():
    assert client.post("/api/v1/voice/speech", json={"text": " "}).status_code == 422
    assert client.post("/api/v1/voice/speech", json={"text": "x" * 1201}).status_code == 422


def test_speech_rejects_invalid_provider_audio(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(200, json={"unexpected": "json"})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/speech", json={"text": "Read this field note."})

    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "voice_unknown"
    assert reply.json()["error"]["stage"] == "tts"


def test_speech_reports_missing_voice_configuration(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(404, text="private provider detail")
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/speech", json={"text": "Read this field note."})

    assert reply.status_code == 404
    assert reply.json()["error"]["code"] == "voice_not_found"
    assert "private provider detail" not in reply.text


def test_malformed_agent_token_response_is_structured(monkeypatch):
    configured(monkeypatch)
    response = httpx.Response(200, json={"unexpected": "private-provider-data"})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "voice_unknown"
    assert reply.json()["error"]["stage"] == "agent_token"
    assert "private-provider-data" not in reply.text


def test_speech_error_mapping_and_logs_never_include_secrets(monkeypatch, caplog):
    configured(monkeypatch)
    secret_key = voice.settings.api_key
    response = httpx.Response(402, json={"detail": {"status": "paid_plan_required", "message": secret_key}})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/speech", json={"text": "Test."})

    assert reply.status_code == 402
    assert reply.json()["error"]["code"] == "voice_plan_required"
    assert reply.json()["error"]["stage"] == "tts"
    assert secret_key not in caplog.text
    assert secret_key not in reply.text


def test_provider_error_categories_are_mapped_without_exposing_provider_body(monkeypatch):
    configured(monkeypatch)
    for status_code, provider_status, expected_code in (
        (403, "detected_unusual_activity", "voice_blocked"),
        (404, "voice_not_found", "voice_not_found"),
        (422, "other", "voice_bad_request"),
        (429, "quota_exceeded", "voice_rate_limited"),
    ):
        response = httpx.Response(status_code, json={"detail": {"status": provider_status, "message": "private-provider-response"}})
        monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))
        reply = client.post("/api/v1/voice/speech", json={"text": "Test."})

        assert reply.status_code == status_code
        assert reply.json()["error"]["code"] == expected_code
        assert reply.json()["error"]["stage"] == "tts"
        assert "private-provider-response" not in reply.text


def test_mapped_provider_errors_are_structured_on_both_voice_routes(monkeypatch):
    configured(monkeypatch)
    cases = (
        (401, "invalid_api_key", "voice_auth_failed"),
        (403, "missing_permissions", "voice_permission_missing"),
        (402, "paid_plan_required", "voice_plan_required"),
        (404, "not_found", "voice_not_found"),
        (400, "other", "voice_bad_request"),
        (422, "other", "voice_bad_request"),
        (429, "quota_exceeded", "voice_rate_limited"),
        (403, "detected_unusual_activity", "voice_blocked"),
        (500, "other", "voice_unknown"),
    )
    for endpoint in ("/api/v1/voice/session", "/api/v1/voice/speech"):
        for status_code, provider_status, code in cases:
            response = httpx.Response(status_code, json={"detail": {"status": provider_status, "message": "must-not-escape"}})
            monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))
            request = {} if endpoint.endswith("/session") else {"text": "Test."}

            reply = client.post(endpoint, json=request)

            assert reply.status_code == (502 if code == "voice_unknown" else status_code)
            assert reply.json()["error"]["code"] == code
            assert reply.json()["error"]["stage"] == ("agent_token" if endpoint.endswith("/session") else "tts")
            assert "must-not-escape" not in reply.text


def test_network_failures_are_structured_for_both_voice_routes(monkeypatch, caplog):
    configured(monkeypatch)
    secret_key = voice.settings.api_key
    for endpoint in ("/api/v1/voice/session", "/api/v1/voice/speech"):
        monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: RaisingAsyncClient(error=httpx.ConnectError(secret_key)))
        request = {} if endpoint.endswith("/session") else {"text": "Test."}

        reply = client.post(endpoint, json=request)

        assert reply.status_code == 502
        assert reply.json()["error"]["code"] == "voice_unreachable"
        assert reply.json()["error"]["stage"] == ("agent_token" if endpoint.endswith("/session") else "tts")
    assert secret_key not in caplog.text


def test_session_success_does_not_log_short_lived_token(monkeypatch, caplog):
    configured(monkeypatch)
    private_token = "a-very-private-short-lived-token"
    response = httpx.Response(200, json={"token": private_token})
    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient(response=response))

    reply = client.post("/api/v1/voice/session", json={})

    assert reply.status_code == 200
    assert private_token == reply.json()["conversation_token"]
    assert private_token not in caplog.text
