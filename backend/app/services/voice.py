from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class VoiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BACKEND_ENV_FILE), env_prefix="ELEVENLABS_", extra="ignore")

    api_key: str = ""
    agent_id: str = ""
    voice_id: str = ""


settings = VoiceSettings()
