"""Configuración central del microservicio ML."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Variables de entorno cargadas con pydantic-settings."""

    nexora_api_url: str = "http://localhost:3000"
    nexora_api_token: str = "nexora-ml-service-token"
    ml_port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
