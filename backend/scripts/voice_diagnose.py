"""Run a sanitized, one-shot connectivity check for the configured ElevenLabs services."""

from __future__ import annotations

import argparse
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes.voice import ALLOWED_PROVIDER_STATUSES
from app.services.voice import settings

HOST = "api.elevenlabs.io"
BASE_URL = f"https://{HOST}/v1"


def machine_status(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (ValueError, TypeError):
        return "other"
    detail = data.get("detail") if isinstance(data, dict) else None
    status = detail.get("status") if isinstance(detail, dict) else None
    return status if isinstance(status, str) and status in ALLOWED_PROVIDER_STATUSES else "other"


def mapped_code(response: httpx.Response) -> str:
    status = machine_status(response)
    if status == "invalid_api_key" or response.status_code == 401:
        return "voice_auth_failed"
    if status == "missing_permissions" or response.status_code == 403:
        return "voice_permission_missing"
    if status == "paid_plan_required" or response.status_code == 402:
        return "voice_plan_required"
    if status == "detected_unusual_activity":
        return "voice_blocked"
    if status in {"voice_not_found", "not_found"} or response.status_code == 404:
        return "voice_not_found"
    if status == "quota_exceeded" or response.status_code == 429:
        return "voice_rate_limited"
    if response.status_code in (400, 422):
        return "voice_bad_request"
    if response.is_success:
        return "success"
    return "voice_unknown"


def check_reachability() -> None:
    try:
        addresses = socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM)
        with socket.create_connection((addresses[0][4][0], 443), timeout=8) as raw_socket:
            with ssl.create_default_context().wrap_socket(raw_socket, server_hostname=HOST):
                pass
        print("reachability: dns=ok tls=ok")
    except socket.gaierror:
        print("reachability: dns=failed tls=not_checked")
    except (OSError, ssl.SSLError, TimeoutError) as exc:
        print(f"reachability: dns=ok tls=failed error_class={type(exc).__name__}")


def check_tts(client: httpx.Client, api_key: str, voice_id: str, label: str) -> None:
    if not api_key or not voice_id:
        print(f"tts[{label}]: skipped reason=config_missing")
        return
    try:
        response = client.post(
            f"{BASE_URL}/text-to-speech/{quote(voice_id, safe='')}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            json={"text": "Test.", "model_id": "eleven_multilingual_v2"},
        )
    except httpx.RequestError as exc:
        print(f"tts[{label}]: mapped_code=voice_unreachable error_class={type(exc).__name__}")
        return
    if response.is_success and response.content and response.headers.get("content-type", "").lower().startswith("audio/"):
        print(f"tts[{label}]: status={response.status_code} mapped_code=success content_type={response.headers.get('content-type')} bytes={len(response.content)}")
    else:
        print(f"tts[{label}]: status={response.status_code} mapped_code={mapped_code(response)} provider_status={machine_status(response)}")


def check_agent_token(client: httpx.Client, api_key: str, agent_id: str) -> None:
    if not api_key or not agent_id:
        print("agent_token: skipped reason=config_missing")
        return
    try:
        response = client.get(
            f"{BASE_URL}/convai/conversation/token",
            params={"agent_id": agent_id},
            headers={"xi-api-key": api_key},
        )
    except httpx.RequestError as exc:
        print(f"agent_token: mapped_code=voice_unreachable error_class={type(exc).__name__}")
        return
    token_exists = False
    if response.is_success:
        try:
            data = response.json()
            token_exists = isinstance(data, dict) and isinstance(data.get("token"), str) and bool(data["token"])
        except (ValueError, TypeError):
            pass
    print(f"agent_token: status={response.status_code} mapped_code={mapped_code(response)} provider_status={machine_status(response)} token_field_exists={token_exists} endpoint_contract=verified_from_installed_sdk")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-id", help="Optionally run one extra TTS check using a supplied voice ID.")
    args = parser.parse_args()

    check_reachability()
    with httpx.Client(timeout=httpx.Timeout(12.0, connect=5.0), follow_redirects=False) as client:
        check_tts(client, settings.api_key, settings.voice_id, "configured")
        if args.voice_id:
            check_tts(client, settings.api_key, args.voice_id, "provided")
        check_agent_token(client, settings.api_key, settings.agent_id)


if __name__ == "__main__":
    main()
