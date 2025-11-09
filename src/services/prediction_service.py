"""
Servicio de Predicción
Responsabilidad: SOLO realizar predicciones usando el modelo de MLflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from ..core.mlflow_client import MLflowClient
from ..utils.data_validator import prepare_dataframe, validate_features

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Servicio de predicción de calidad de vino
    Responsabilidad: Realizar predicciones individuales y por lote
    """

    def __init__(self, mlflow_client: MLflowClient):
        """
        Inicializa el servicio de predicción

        Args:
            mlflow_client: Cliente de MLflow configurado
        """
        self.mlflow_client = mlflow_client

        # Cargar modelo
        self.model = mlflow_client.load_model()

        # Obtener información del modelo
        self.model_info = mlflow_client.get_model_info()

        logger.info("Prediction Service initialized")

    def predict_single(self, wine_features: Dict[str, float]) -> Dict:
        try:
            logger.info("Processing single prediction...")

            # Validar features
            validation = validate_features(wine_features)
            if not validation['is_valid']:
                raise ValueError(
                    f"Missing required features: {validation['missing_features']}"
                )

            wine_features_float = {
                key: float(value) for key, value in wine_features.items()
            }

            # Preparar DataFrame
            df = prepare_dataframe(wine_features)

            df = df.astype('float64')

            # Predecir
            prediction = int(self.model.predict(df)[0])

            # Obtener probabilidades
            probabilities_dict, confidence = self._get_probabilities(df, prediction)

            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities_dict,
                'features': wine_features_float,
                'model_info': self.model_info
            }

            logger.info(f"Prediction completed: {prediction}")

            return result

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise Exception(f"Failed to make prediction: {str(e)}")

    def predict_batch(self, csv_file_path: str) -> Dict:
        """
        Realiza predicciones para múltiples vinos desde CSV

        Args:
            csv_file_path: Ruta al archivo CSV con los datos
                El CSV debe tener las 11 columnas requeridas

        Returns:
            Diccionario con:
                - dataframe: DataFrame con predicciones agregadas
                - predictions: Lista de predicciones (0s y 1s)
                - summary: Estadísticas del lote
                - model_info: Información del modelo

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el CSV tiene formato incorrecto
        """
        try:
            logger.info(f"Processing batch from: {csv_file_path}")

            # Leer CSV
            df = pd.read_csv(csv_file_path, sep=";")

            total_rows = len(df)
            logger.info(f"Read {total_rows} rows from CSV")

            # Preparar datos (valida y ordena columnas)
            X = prepare_dataframe(df)

            # Realizar predicciones
            predictions = self.model.predict(X)
            predictions = predictions.astype(int)

            # Agregar predicciones al DataFrame original
            df_result = df.copy()
            df_result['prediction'] = predictions
            df_result['prediction_label'] = [
                'Alta Calidad' if p == 1 else 'Baja Calidad'
                for p in predictions
            ]

            # Calcular estadísticas
            summary = self._calculate_batch_statistics(predictions)
            summary['model_info'] = self.model_info

            logger.info(
                f"Batch completed: "
                f"{summary['high_quality_count']} high, "
                f"{summary['low_quality_count']} low"
            )

            return {
                'dataframe': df_result,
                'predictions': predictions.tolist(),
                'summary': summary
            }

        except FileNotFoundError as e:
            logger.error(f"File not found: {csv_file_path}")
            raise
        except ValueError as e:
            logger.error(f"Invalid CSV format: {e}")
            raise
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise Exception(f"Failed to process batch: {str(e)}")

    def _get_probabilities(
            self,
            df: pd.DataFrame,
            prediction: int
    ) -> tuple[Optional[Dict], Optional[float]]:
        """
        Intenta obtener probabilidades del modelo

        Args:
            df: DataFrame con features
            prediction: Predicción realizada

        Returns:
            Tupla con (dict_probabilidades, confianza)
        """
        try:
            # Cargar modelo sklearn para obtener probabilidades
            import mlflow.sklearn

            model_uri = f"models:/{self.mlflow_client.model_name}/{self.mlflow_client.stage}"
            sklearn_model = mlflow.sklearn.load_model(model_uri)

            if hasattr(sklearn_model, 'predict_proba'):
                proba_array = sklearn_model.predict_proba(df)[0]

                probabilities_dict = {
                    'Baja Calidad': float(proba_array[0]),
                    'Alta Calidad': float(proba_array[1])
                }

                confidence = float(proba_array[prediction])

                return probabilities_dict, confidence
            else:
                logger.warning("Model does not support predict_proba")
                return None, None

        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
            return None, None

    def _calculate_batch_statistics(self, predictions: np.ndarray) -> Dict:
        """
        Calcula estadísticas de un lote de predicciones

        Args:
            predictions: Array con predicciones

        Returns:
            Diccionario con estadísticas
        """
        total = len(predictions)
        high_count = int(np.sum(predictions == 1))
        low_count = int(np.sum(predictions == 0))

        high_pct = (high_count / total) * 100 if total > 0 else 0
        low_pct = (low_count / total) * 100 if total > 0 else 0

        return {
            'total_samples': total,
            'high_quality_count': high_count,
            'low_quality_count': low_count,
            'high_quality_percentage': high_pct,
            'low_quality_percentage': low_pct
        }

    def get_model_metadata(self) -> Dict:
        """
        Obtiene metadata del modelo cargado

        Returns:
            Diccionario con información del modelo
        """
        return {
            'model_name': self.mlflow_client.model_name,
            'stage': self.mlflow_client.stage,
            'tracking_uri': self.mlflow_client.tracking_uri,
            **self.model_info
        }