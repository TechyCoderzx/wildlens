import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.vision import router as vision_router
from app.routes.voice import router as voice_router
from app.services.voice import settings as voice_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "ElevenLabs configuration presence: elevenlabs_key_loaded=%s voice_id_loaded=%s agent_id_loaded=%s",
        bool(voice_settings.api_key.strip()),
        bool(voice_settings.voice_id.strip()),
        bool(voice_settings.agent_id.strip()),
    )
    yield


app = FastAPI(
    title="WildLens API",
    description="Local-first nature identification API. Model results are hypotheses, not verified scientific identifications.",
    version="0.1.0",
    lifespan=lifespan,
)

extra_cors_origins = [
    origin.strip()
    for origin in os.getenv("WILDLENS_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *extra_cors_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(vision_router, prefix="/api/v1/vision", tags=["vision"])
app.include_router(voice_router, prefix="/api/v1/voice", tags=["voice"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "wildlens-api"}
