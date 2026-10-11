from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.vision import IdentificationResult
from app.services.voice import settings

router = APIRouter()
logger = logging.getLogger(__name__)
ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ALLOWED_PROVIDER_STATUSES = {
    "invalid_api_key",
    "missing_permissions",
    "paid_plan_required",
    "quota_exceeded",
    "detected_unusual_activity",
    "voice_not_found",
    "not_found",
}
PUBLIC_ERRORS = {
    "voice_not_configured": "Voice isn't configured on the server.",
    "voice_auth_failed": "The ElevenLabs key was rejected.",
    "voice_permission_missing": "The ElevenLabs key lacks permission for this feature.",
    "voice_plan_required": "This voice isn't available on the current ElevenLabs plan. Choose a default voice.",
    "voice_not_found": "The configured voice or agent was not found.",
    "voice_bad_request": "The voice request was invalid.",
    "voice_rate_limited": "Voice service is busy or the quota is used up.",
    "voice_blocked": "ElevenLabs blocked this request (free-plan restrictions, possibly VPN or proxy).",
    "voice_unreachable": "ElevenLabs could not be reached. Check the backend network connection and try again.",
    "voice_unknown": "ElevenLabs could not complete the voice request. Check the server logs for its sanitized failure stage.",
}


class VoiceSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identification: IdentificationResult | None = None


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1200)


class VoiceFailure(Exception):
    def __init__(self, status_code: int, code: str, stage: str):
        self.status_code = status_code
        self.code = code
        self.stage = stage

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={"error": {"code": self.code, "stage": self.stage, "message": PUBLIC_ERRORS[self.code]}},
        )


def build_agent_prompt(identification: IdentificationResult | None) -> str:
    context = "No identification is currently open. Help with WildLens usage and invite the user to identify a nature photograph."
    if identification is not None:
        data = {
            "status": identification.status,
            "common_name": identification.common_name,
            "scientific_name": identification.scientific_name,
            "confidence": identification.confidence,
            "summary": identification.summary,
            "visible_evidence": identification.visible_evidence[:5],
            "evidence_details": [
                {"feature": item.feature, "inference": item.inference, "support": item.support, "limitation": item.limitation, "source": item.source}
                for item in identification.evidence_details[:4]
            ],
            "limitations": identification.evidence_assessment.limitations[:4],
            "candidates": [
                {
                    "common_name": candidate.common_name,
                    "scientific_name": candidate.scientific_name,
                    "match_explanation": candidate.match_explanation,
                    "supporting_features": candidate.supporting_features[:2],
                    "distinguishing_features": [
                        {"feature": feature.feature, "observation": feature.observation, "interpretation": feature.interpretation}
                        for feature in candidate.distinguishing_features[:2]
                    ],
                    "uncertainties": candidate.uncertainties[:2],
                }
                for candidate in identification.candidates[:3]
            ],
            "observe_next": identification.observe_next,
            "observation_tasks": [
                {"title": task.title, "what_to_look_for": task.what_to_look_for, "why_it_matters": task.why_it_matters, "how_safely": task.how_safely}
                for task in identification.observation_tasks[:3]
            ],
        }
        context = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are WildLens, a concise and careful nature field-guide assistant. "
        "Treat the field-note context below as the only case-specific evidence. "
        "Distinguish image-visible evidence from user-reported observations. "
        "Keep the identification tentative at the confidence and status supplied; never invent facts, "
        "species, evidence, measurements, or certainty. If the note is unknown or uncertain, say so. "
        "Suggest safe observation from a respectful distance. Do not expose internal reasoning; give short conclusions.\n\n"
        f"Current field note (validated JSON data):\n{context}"
    )


def failure(code: str, stage: str, status_code: int) -> VoiceFailure:
    return VoiceFailure(status_code, code, stage)


def require_credentials(*, stage: str, need_agent: bool = True, need_voice: bool = False) -> tuple[str, str, str]:
    if not settings.api_key or (need_agent and not settings.agent_id) or (need_voice and not settings.voice_id):
        # Never include missing values or settings contents in the diagnostic log.
        logger.warning("ElevenLabs request failed stage=%s category=config_missing", stage)
        raise failure("voice_not_configured", "config", 503)
    return settings.api_key, settings.agent_id, settings.voice_id


def provider_status(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (ValueError, TypeError):
        return "other"
    detail = data.get("detail") if isinstance(data, dict) else None
    value = detail.get("status") if isinstance(detail, dict) else None
    return value if isinstance(value, str) and value in ALLOWED_PROVIDER_STATUSES else "other"


def provider_failure(response: httpx.Response, stage: str) -> VoiceFailure:
    status = response.status_code
    category = provider_status(response)
    logger.warning("ElevenLabs request failed stage=%s upstream_status=%s provider_status=%s", stage, status, category)
    if category == "invalid_api_key":
        return failure("voice_auth_failed", stage, 401)
    if category == "missing_permissions":
        return failure("voice_permission_missing", stage, 403)
    if category == "paid_plan_required" or status == 402:
        return failure("voice_plan_required", stage, 402)
    if category == "detected_unusual_activity":
        return failure("voice_blocked", stage, 403)
    if category in {"voice_not_found", "not_found"} or status == 404:
        return failure("voice_not_found", stage, 404)
    if category == "quota_exceeded" or status == 429:
        return failure("voice_rate_limited", stage, 429)
    if status == 401:
        return failure("voice_auth_failed", stage, 401)
    if status == 403:
        return failure("voice_permission_missing", stage, 403)
    if status in (400, 422):
        return failure("voice_bad_request", stage, status)
    return failure("voice_unknown", stage, 502)


def log_network_failure(stage: str, exc: httpx.RequestError) -> VoiceFailure:
    logger.warning("ElevenLabs request failed stage=%s category=network error_class=%s", stage, type(exc).__name__)
    return failure("voice_unreachable", stage, 504 if isinstance(exc, httpx.TimeoutException) else 502)


@router.post("/session")
async def create_session(request: VoiceSessionRequest):
    try:
        api_key, agent_id, _ = require_credentials(stage="agent_token")
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            response = await client.get(
                f"{ELEVENLABS_BASE}/convai/conversation/token",
                params={"agent_id": agent_id},
                headers={"xi-api-key": api_key},
            )
        if not response.is_success:
            raise provider_failure(response, "agent_token")
        try:
            token = response.json().get("token")
        except (ValueError, AttributeError):
            token = None
        if not isinstance(token, str) or not token:
            logger.warning("ElevenLabs request failed stage=agent_token upstream_status=%s category=malformed_response", response.status_code)
            raise failure("voice_unknown", "agent_token", 502)
        return {"conversation_token": token, "agent_prompt": build_agent_prompt(request.identification)}
    except VoiceFailure as exc:
        return exc.response()
    except httpx.RequestError as exc:
        return log_network_failure("agent_token", exc).response()


@router.post("/speech")
async def create_speech(request: SpeechRequest):
    text = request.text.strip()
    if not text:
        return failure("voice_bad_request", "tts", 422).response()
    try:
        api_key, _, voice_id = require_credentials(stage="tts", need_agent=False, need_voice=True)
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
            response = await client.post(
                f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                json={"text": text, "model_id": "eleven_multilingual_v2"},
            )
        if not response.is_success:
            raise provider_failure(response, "tts")
        if not response.content or not response.headers.get("content-type", "").lower().startswith("audio/"):
            logger.warning("ElevenLabs request failed stage=tts upstream_status=%s category=invalid_audio_response", response.status_code)
            raise failure("voice_unknown", "tts", 502)
        return Response(content=response.content, media_type=response.headers.get("content-type", "audio/mpeg"), headers={"Cache-Control": "no-store"})
    except VoiceFailure as exc:
        return exc.response()
    except httpx.RequestError as exc:
        return log_network_failure("tts", exc).response()
