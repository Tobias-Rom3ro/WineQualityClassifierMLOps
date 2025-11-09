"""
Servicio de Interpretación con IA Generativa
Responsabilidad: SOLO generar explicaciones en lenguaje natural
"""

from typing import Dict, Optional
import logging

from ..core.genai_client import GenAIClient
from ..utils.formatters import format_features_for_display

logger = logging.getLogger(__name__)


class InterpretationService:
    """
    Servicio de interpretación de predicciones usando GenAI
    Responsabilidad: Generar explicaciones en lenguaje natural
    """

    def __init__(self, genai_client: GenAIClient):
        """
        Inicializa el servicio de interpretación

        Args:
            genai_client: Cliente de GenAI configurado
        """
        self.genai_client = genai_client

        logger.info("Interpretation Service initialized")

    def interpret_single_prediction(
            self,
            prediction_result: Dict,
            include_confidence: bool = True
    ) -> str:
        """
        Genera explicación para una predicción individual

        Args:
            prediction_result: Resultado de PredictionService.predict_single()
                Debe contener: prediction, features, confidence (opcional)
            include_confidence: Si incluir información de confianza

        Returns:
            Explicación en lenguaje natural generada por IA
        """
        try:
            logger.info("Generating single prediction interpretation...")

            # Extraer información
            prediction = prediction_result['prediction']
            features = prediction_result['features']
            confidence = prediction_result.get('confidence')

            # Formatear características
            features_text = format_features_for_display(features)

            # Determinar etiqueta
            prediction_label = "Alta Calidad (≥6)" if prediction == 1 else "Baja Calidad (<6)"

            # Información de confianza
            confidence_info = ""
            if include_confidence and confidence is not None:
                confidence_pct = confidence * 100
                confidence_info = f"\nConfianza del modelo: {confidence_pct:.1f}%"

            # Construir prompt
            prompt = self._build_single_prediction_prompt(
                features_text,
                prediction_label,
                confidence_info
            )

            # Generar explicación
            explanation = self.genai_client.generate(prompt)

            logger.info("Interpretation generated")

            return explanation

        except Exception as e:
            logger.error(f"❌ Failed to generate interpretation: {e}")
            return "No se pudo generar una explicación para esta predicción."

    def interpret_batch_predictions(self, summary: Dict) -> str:
        """
        Genera análisis para predicciones por lote

        Args:
            summary: Resumen de PredictionService.predict_batch()
                Debe contener: total_samples, high_quality_count,
                              low_quality_count, percentages

        Returns:
            Análisis en lenguaje natural generado por IA
        """
        try:
            logger.info("Generating batch interpretation...")

            # Extraer estadísticas
            total = summary['total_samples']
            high_count = summary['high_quality_count']
            low_count = summary['low_quality_count']
            high_pct = summary['high_quality_percentage']
            low_pct = summary['low_quality_percentage']

            # Construir prompt
            prompt = self._build_batch_analysis_prompt(
                total, high_count, low_count, high_pct, low_pct
            )

            # Generar análisis
            analysis = self.genai_client.generate(prompt)

            logger.info("Batch interpretation generated")

            return analysis

        except Exception as e:
            logger.error(f"❌ Failed to generate batch interpretation: {e}")
            return "No se pudo generar un análisis para este lote."

    def explain_feature_importance(
            self,
            features: Dict[str, float],
            prediction_label: str,
            top_features: Optional[list] = None
    ) -> str:
        """
        Explica la importancia de features específicas

        Args:
            features: Características del vino
            prediction_label: Etiqueta de predicción
            top_features: Features más importantes (opcional)

        Returns:
            Explicación enfocada en features importantes
        """
        try:
            # Formatear features (destacando las importantes)
            features_text = format_features_for_display(features, top_features)

            # Construir prompt
            prompt = self._build_feature_importance_prompt(
                features_text,
                prediction_label,
                top_features
            )

            # Generar explicación
            explanation = self.genai_client.generate(prompt)

            return explanation

        except Exception as e:
            logger.error(f"❌ Failed to explain importance: {e}")
            return "No se pudo generar explicación de importancia."

    def get_wine_profile(self, features: Dict[str, float]) -> str:
        """
        Genera descripción del perfil organoléptico esperado

        Args:
            features: Características del vino

        Returns:
            Descripción del perfil del vino
        """
        try:
            features_text = format_features_for_display(features)

            prompt = f"""
Como sommelier profesional, describe brevemente (2-3 líneas) el perfil 
organoléptico esperado de un vino blanco con estas características:

{features_text}

Menciona: cuerpo, acidez percibida, dulzor, y posibles notas aromáticas.
Usa lenguaje de cata profesional pero comprensible.
"""

            description = self.genai_client.generate(prompt)

            return description

        except Exception as e:
            logger.error(f"❌ Failed to generate wine profile: {e}")
            return "No se pudo generar descripción del perfil."

    # ========================================================================
    # MÉTODOS PRIVADOS PARA CONSTRUCCIÓN DE PROMPTS
    # ========================================================================

    def _build_single_prediction_prompt(
            self,
            features_text: str,
            prediction_label: str,
            confidence_info: str
    ) -> str:
        """Construye prompt para predicción individual"""
        return f"""
Eres un sommelier experto y enólogo profesional. Un modelo de machine learning 
ha analizado las características fisicoquímicas de un vino blanco y ha realizado 
una clasificación.

CARACTERÍSTICAS DEL VINO ANALIZADO:
{features_text}

PREDICCIÓN DEL MODELO: {prediction_label}
{confidence_info}

Tu tarea es proporcionar una explicación profesional y comprensible de 3-4 líneas que:
1. Explique POR QUÉ el modelo clasificó este vino con esa calidad
2. Identifique las 2-3 características químicas MÁS DETERMINANTES para esta predicción
3. Proporcione una breve recomendación sobre el perfil organoléptico esperado

Sé técnico pero comprensible, directo y conciso. Usa terminología enológica apropiada.
No menciones el modelo de ML, enfócate en las características del vino.
"""

    def _build_batch_analysis_prompt(
            self,
            total: int,
            high_count: int,
            low_count: int,
            high_pct: float,
            low_pct: float
    ) -> str:
        """Construye prompt para análisis de lote"""
        return f"""
Eres un enólogo y analista de calidad de vinos. Has recibido los resultados de 
un análisis masivo de vinos blancos realizado por un sistema de machine learning.

RESULTADOS DEL ANÁLISIS:
- Total de vinos analizados: {total}
- Vinos de ALTA calidad (≥6/10): {high_count} ({high_pct:.1f}%)
- Vinos de BAJA calidad (<6/10): {low_count} ({low_pct:.1f}%)

Tu tarea es proporcionar un análisis profesional de 3-4 líneas que:
1. Interprete QUÉ INDICA esta distribución sobre la calidad general del lote
2. Identifique si hay alguna TENDENCIA notable (dominancia de alta/baja calidad)
3. Proporcione una CONCLUSIÓN sobre el lote y su potencial comercial

Sé profesional, objetivo y conciso. Usa terminología enológica apropiada.
"""

    def _build_feature_importance_prompt(
            self,
            features_text: str,
            prediction_label: str,
            top_features: Optional[list]
    ) -> str:
        """Construye prompt para explicación de importancia"""
        focus = ""
        if top_features:
            focus = f"\nEnfócate especialmente en: {', '.join(top_features)}"

        return f"""
Como sommelier experto, explica de manera muy breve (2 líneas) cómo las 
siguientes características químicas contribuyeron a que este vino sea 
clasificado como {prediction_label}.

CARACTERÍSTICAS:
{features_text}
{focus}

Menciona solo las 2 características MÁS relevantes y su efecto en la calidad.
"""

    def test_connection(self) -> bool:
        """
        Prueba la conexión con el proveedor de GenAI

        Returns:
            True si la conexión es exitosa
        """
        return self.genai_client.test_connection()