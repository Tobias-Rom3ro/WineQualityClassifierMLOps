"""
Wine Quality MLOps - Módulo principal
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"

# Importar componentes principales
from .config import get_mlflow_config, get_genai_config, get_app_config
from .core import MLflowClient, GenAIClient
from .services import PredictionService, InterpretationService
from .ui import GradioApp

__all__ = [
    # Config
    'get_mlflow_config',
    'get_genai_config',
    'get_app_config',

    # Core
    'MLflowClient',
    'GenAIClient',

    # Services
    'PredictionService',
    'InterpretationService',

    # UI
    'GradioApp'
]