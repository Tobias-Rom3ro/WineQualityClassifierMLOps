"""
Configuración centralizada del proyecto
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://127.0.0.1:5000"
    model_name: str = "wine-quality-classifier"
    default_stage: str = "Production"

    @classmethod
    def from_env(cls):
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            model_name=os.getenv("MODEL_NAME", "wine-quality-classifier"),
            default_stage=os.getenv("MODEL_STAGE", "Production")
        )


@dataclass
class GenAIConfig:
    provider: str = "gemini"
    api_key: Optional[str] = None
    model_name: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def from_env(cls):
        return cls(
            provider=os.getenv("GENAI_PROVIDER", "gemini"),
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GENAI_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("GENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("GENAI_MAX_TOKENS", "1024"))
        )


@dataclass
class AppConfig:
    app_name: str = "Wine Quality Predictor"
    version: str = "1.0.0"
    debug: bool = False
    server_host: str = "0.0.0.0"
    server_port: int = 7860

    @classmethod
    def from_env(cls):
        return cls(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("SERVER_PORT", "7860"))
        )

_mlflow_config = None
_genai_config = None
_app_config = None

def get_mlflow_config() -> MLflowConfig:
    global _mlflow_config
    if _mlflow_config is None:
        _mlflow_config = MLflowConfig.from_env()
    return _mlflow_config


def get_genai_config() -> GenAIConfig:
    global _genai_config
    if _genai_config is None:
        _genai_config = GenAIConfig.from_env()
    return _genai_config


def get_app_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig.from_env()
    return _app_config