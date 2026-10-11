import json
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.schemas.vision import FollowUpInput, FollowUpResponse, IdentificationResult
from app.services.ollama import ollama_status
from app.services.vision import analyze_follow_up, analyze_image

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.get("/status")
async def status():
    return await ollama_status()


@router.post("/analyze", response_model=IdentificationResult)
async def analyze(image: UploadFile = File(...)):
    request_started = time.perf_counter()
    logger.info("vision_timing event=request_received")
    try:
        image_read_started = time.perf_counter()
        content = await read_image(image)
        logger.info("vision_timing stage=image_upload_read duration_ms=%.1f", (time.perf_counter() - image_read_started) * 1000)
        model_started = time.perf_counter()
        try:
            return await analyze_image(content, image.content_type)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"Model returned an invalid response: {exc}") from exc
        finally:
            logger.info("vision_timing stage=model_service duration_ms=%.1f", (time.perf_counter() - model_started) * 1000)
    finally:
        logger.info("vision_timing stage=backend_route_total duration_ms=%.1f", (time.perf_counter() - request_started) * 1000)


@router.post("/follow-up", response_model=FollowUpResponse)
async def follow_up(
    previous: str = Form(..., max_length=20_000),
    observation: str | None = Form(default=None, max_length=800),
    image: UploadFile | None = File(default=None),
):
    if image is None and not (observation and observation.strip()):
        raise HTTPException(status_code=422, detail="Add an observation, a follow-up photo, or both.")
    try:
        prior = IdentificationResult.model_validate_json(previous)
        request = FollowUpInput(previous=prior, observation=observation)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Previous identification or observation is invalid.") from exc

    content = await read_image(image) if image is not None else None
    try:
        updated = await analyze_follow_up(
            content,
            image.content_type if image is not None else None,
            request.previous,
            request.observation.strip() if request.observation else None,
        )
        return FollowUpResponse(
            identification=updated,
            what_changed=updated.change_summary,
            remaining_uncertainties=updated.evidence_assessment.limitations,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned an invalid response: {exc}") from exc


async def read_image(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    return content
