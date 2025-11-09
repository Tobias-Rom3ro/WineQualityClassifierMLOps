"""
Cliente de MLflow - Abstracción para cargar modelos
"""

import mlflow
import mlflow.pyfunc
from typing import Dict, Optional
import logging

from ..config.settings import get_mlflow_config

logger = logging.getLogger(__name__)


class MLflowClient:
    def __init__(
            self,
            tracking_uri: Optional[str] = None,
            model_name: Optional[str] = None,
            stage: Optional[str] = None
    ):
        config = get_mlflow_config()

        self.tracking_uri = tracking_uri or config.tracking_uri
        self.model_name = model_name or config.model_name
        self.stage = stage or config.default_stage

        # Configurar MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        logger.info(f"MLflow Client initialized: {self.tracking_uri}")

    def load_model(self, model_name: Optional[str] = None, stage: Optional[str] = None):
        """
        Carga modelo desde MLflow Registry

        Args:
            model_name: Nombre del modelo (usa self.model_name si es None)
            stage: Stage (usa self.stage si es None)

        Returns:
            Modelo cargado

        Raises:
            Exception: Si no se puede cargar el modelo
        """
        model_name = model_name or self.model_name
        stage = stage or self.stage

        try:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model: {model_uri}")

            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(f" Model cargado satisfactoriamente: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Falló cargar modelo: {e}")
            raise Exception(
                f"Could not load model '{model_name}' from stage '{stage}'. "
            )

    def get_model_info(self, model_name: Optional[str] = None, stage: Optional[str] = None) -> Dict:
        """
        Obtiene información del modelo desde Registry

        Args:
            model_name: Nombre del modelo
            stage: Stage del modelo

        Returns:
            Diccionario con información del modelo
        """
        model_name = model_name or self.model_name
        stage = stage or self.stage

        try:
            client = mlflow.tracking.MlflowClient()

            # Buscar versiones del modelo
            versions = client.search_model_versions(f"name='{model_name}'")

            # Buscar versión en el stage especificado
            for version in versions:
                if version.current_stage == stage:
                    return {
                        "name": model_name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "status": version.status,
                        "creation_timestamp": version.creation_timestamp
                    }

            # Si no se encuentra, retornar info básica
            return {
                "name": model_name,
                "stage": stage,
                "warning": f"No version found in stage '{stage}'"
            }

        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {
                "name": model_name,
                "stage": stage,
                "error": str(e)
            }

    def list_models(self) -> list:
        """
        Lista todos los modelos registrados

        Returns:
            Lista de modelos
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_versions(self, model_name: Optional[str] = None) -> list:
        """
        Obtiene todas las versiones de un modelo

        Args:
            model_name: Nombre del modelo

        Returns:
            Lista de versiones
        """
        model_name = model_name or self.model_name

        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")

            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []