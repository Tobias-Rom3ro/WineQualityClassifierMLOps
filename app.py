"""
Punto de entrada principal de la aplicación
Wine Quality Predictor - MLOps Project

Este script inicializa todos los servicios y lanza la aplicación Gradio.
"""

import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_mlflow_config, get_genai_config, get_app_config
from src.core import MLflowClient, GenAIClient
from src.services import PredictionService, InterpretationService
from src.ui import GradioApp

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


def validate_environment():
    """
    Valida que el entorno esté correctamente configurado

    Raises:
        EnvironmentError: Si falta alguna configuración crítica
    """
    logger.info(" Validating environment...")

    # Validar API key de GenAI
    genai_config = get_genai_config()
    if not genai_config.api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment. "
            "Please set it before running the app:\n"
            "export GEMINI_API_KEY='your_api_key_here'"
        )

    # Validar que existe el directorio de MLflow
    mlflow_config = get_mlflow_config()
    if mlflow_config.tracking_uri.startswith("file:"):
        mlruns_path = mlflow_config.tracking_uri.replace("file:", "")
        if not os.path.exists(mlruns_path):
            logger.warning(f"MLflow directory not found: {mlruns_path}")
            logger.warning("Make sure you have trained a model first by running the notebook.")

    logger.info("Environment validation passed")


def initialize_services():
    """
    Inicializa todos los servicios de la aplicación

    Returns:
        Tupla con (prediction_service, interpretation_service)
    """
    logger.info("=" * 70)
    logger.info("WINE QUALITY PREDICTOR - Initializing Services")
    logger.info("=" * 70)

    # 1. Inicializar MLflow Client
    logger.info("\n Step 1: Initializing MLflow Client...")
    mlflow_config = get_mlflow_config()
    logger.info(f"   Tracking URI: {mlflow_config.tracking_uri}")
    logger.info(f"   Model Name: {mlflow_config.model_name}")
    logger.info(f"   Stage: {mlflow_config.default_stage}")

    mlflow_client = MLflowClient()

    # 2. Inicializar GenAI Client
    logger.info("\n Step 2: Initializing GenAI Client...")
    genai_config = get_genai_config()
    logger.info(f"   Provider: {genai_config.provider}")
    logger.info(f"   Model: {genai_config.model_name}")

    genai_client = GenAIClient()

    # Probar conexión con GenAI
    logger.info("   Testing GenAI connection...")
    if genai_client.test_connection():
        logger.info("GenAI connection successful")
    else:
        logger.warning("GenAI connection test failed")

    # 3. Inicializar Prediction Service
    logger.info("\n Step 3: Initializing Prediction Service...")
    prediction_service = PredictionService(mlflow_client)

    # Mostrar info del modelo
    model_metadata = prediction_service.get_model_metadata()
    logger.info(f"   Model loaded: {model_metadata.get('model_name')}")
    logger.info(f"   Version: {model_metadata.get('version', 'N/A')}")
    logger.info(f"   Stage: {model_metadata.get('stage', 'N/A')}")

    # 4. Inicializar Interpretation Service
    logger.info("\n Step 4: Initializing Interpretation Service...")
    interpretation_service = InterpretationService(genai_client)

    logger.info("\n All services initialized successfully")
    logger.info("=" * 70)

    return prediction_service, interpretation_service


def main():
    """
    Función principal de la aplicación
    """
    try:
        # Banner de inicio
        print("\n" + "=" * 70)
        print("🍷 WINE QUALITY PREDICTOR - MLOps Project")
        print("=" * 70)
        print("Version: 1.0.0")
        print("Author: MLOps Team")
        print("=" * 70 + "\n")

        # 1. Validar entorno
        validate_environment()

        # 2. Inicializar servicios
        prediction_service, interpretation_service = initialize_services()

        # 3. Crear aplicación Gradio
        logger.info("\n Creating Gradio interface...")
        app = GradioApp(prediction_service, interpretation_service)

        # 4. Obtener configuración de la app
        app_config = get_app_config()

        # 5. Lanzar aplicación
        logger.info("\n Launching application...")
        logger.info(f"   Server: {app_config.server_host}:{app_config.server_port}")
        logger.info(f"   Debug mode: {app_config.debug}")

        print("\n" + "=" * 70)
        print("✅ Application ready!")
        print("=" * 70)
        print(f"📍 Access the app at: http://localhost:{app_config.server_port}")
        print("=" * 70 + "\n")

        app.launch(
            server_name=app_config.server_host,
            server_port=app_config.server_port,
            share=False,
            debug=app_config.debug
        )

    except EnvironmentError as e:
        logger.error(f"\n Environment Error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n Failed to start application: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()