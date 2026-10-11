import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:e4b"
    max_image_mb: int = 10
    ollama_timeout_seconds: float = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


async def ollama_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", [])]
        return {
            "available": True,
            "base_url": settings.ollama_base_url,
            "configured_model": settings.ollama_model,
            "configured_model_installed": any(
                name == settings.ollama_model or name.startswith(settings.ollama_model + ":")
                for name in models
            ),
            "installed_models": models,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "available": False,
            "base_url": settings.ollama_base_url,
            "configured_model": settings.ollama_model,
            "configured_model_installed": False,
            "installed_models": [],
            "message": "Ollama is unreachable. Start Ollama and retry.",
        }
