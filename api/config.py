"""
api/config.py
=============
Configuration settings for SugamGov AI RAG FastAPI Backend.

Reads configuration values from environment variables and .env file.
Adheres strictly to credential isolation:
- Never prints or logs secrets.
- Excludes sensitive fields from string representations and serialization.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class Settings:
    """
    Application settings loaded from environment variables with safe defaults.
    """
    def __init__(self):
        # Server settings
        self.api_host: str = os.getenv("API_HOST", "127.0.0.1")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.api_reload: bool = os.getenv("API_RELOAD", "true").lower() in ("true", "1", "yes")

        # Database settings
        self.database_url: str = os.getenv("DATABASE_URL", "")

        # Gemini settings
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.gemini_generation_model: str = os.getenv(
            "GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"
        )

        # CORS origins (default to localhost frontend dev ports)
        raw_origins = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        )
        self.cors_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

    def __repr__(self) -> str:
        """Safe representation hiding credentials."""
        has_key = bool(self.gemini_api_key)
        has_db = bool(self.database_url)
        return (
            f"Settings(api_host='{self.api_host}', api_port={self.api_port}, "
            f"model='{self.gemini_generation_model}', has_gemini_key={has_key}, "
            f"has_database_url={has_db}, cors_origins={self.cors_origins})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns singleton instance of application settings."""
    return Settings()
