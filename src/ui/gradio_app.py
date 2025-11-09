"""
Interfaz de Usuario con Gradio
Responsabilidad: SOLO presentar interfaz y conectar con servicios
"""

import gradio as gr
import pandas as pd
from typing import Tuple
import logging

from ..services.prediction_service import PredictionService
from ..services.interpretation_service import InterpretationService
from ..utils.data_validator import get_feature_ranges, create_sample_wine
from ..utils.formatters import format_prediction_label, format_confidence

logger = logging.getLogger(__name__)


class GradioApp:
    """
    Aplicación de Gradio para predicción de calidad de vino
    Responsabilidad: Gestionar interfaz de usuario y coordinar servicios
    """

    def __init__(
            self,
            prediction_service: PredictionService,
            interpretation_service: InterpretationService
    ):
        """
        Inicializa la aplicación Gradio

        Args:
            prediction_service: Servicio de predicción
            interpretation_service: Servicio de interpretación
        """
        self.prediction_service = prediction_service
        self.interpretation_service = interpretation_service

        # Obtener rangos de features para sliders
        self.feature_ranges = get_feature_ranges()

        logger.info("Gradio App initialized")

    def _predict_single_handler(
            self,
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
    ) -> Tuple[str, str, str]:
        """
        Handler para predicción individual

        Returns:
            Tupla con (resultado, explicación, info_modelo)
        """
        try:
            logger.info("Processing single prediction from UI...")

            # 1. Preparar features
            wine_features = {
                'fixed acidity': float(fixed_acidity),
                'volatile acidity': float(volatile_acidity),
                'citric acid': float(citric_acid),
                'residual sugar': float(residual_sugar),
                'chlorides': float(chlorides),
                'free sulfur dioxide': float(free_sulfur_dioxide),
                'total sulfur dioxide': float(total_sulfur_dioxide),
                'density': float(density),
                'pH': float(pH),
                'sulphates': float(sulphates),
                'alcohol': float(alcohol)
            }

            # 2. Obtener predicción
            prediction_result = self.prediction_service.predict_single(wine_features)

            # 3. Generar interpretación
            explanation = self.interpretation_service.interpret_single_prediction(
                prediction_result
            )

            # 4. Formatear salida
            prediction = prediction_result['prediction']
            confidence = prediction_result.get('confidence')

            # Resultado principal
            result_text = format_prediction_label(prediction, include_emoji=True)

            # Información del modelo
            model_info = prediction_result.get('model_info', {})
            info_lines = []

            if confidence is not None:
                info_lines.append(f"**Confianza:** {format_confidence(confidence)}")

            if model_info:
                info_lines.append(f"**Modelo:** {model_info.get('name', 'N/A')}")
                info_lines.append(f"**Versión:** {model_info.get('version', 'N/A')}")
                info_lines.append(f"**Stage:** {model_info.get('stage', 'N/A')}")

            info_text = "\n".join(info_lines) if info_lines else "Info no disponible"

            logger.info(f"UI prediction completed")

            return result_text, explanation, info_text

        except Exception as e:
            logger.error(f"UI prediction error: {e}")
            error_msg = f"Error: {str(e)}"
            return "Error en predicción", error_msg, ""

    def _predict_batch_handler(self, csv_file) -> Tuple[pd.DataFrame, str]:
        """
        Handler para predicción por lote

        Returns:
            Tupla con (dataframe_resultados, resumen_análisis)
        """
        try:
            if csv_file is None:
                return pd.DataFrame(), "⚠️ Por favor, sube un archivo CSV"

            logger.info(f"Processing batch from UI: {csv_file.name}")

            # 1. Obtener predicciones
            batch_result = self.prediction_service.predict_batch(csv_file.name)

            # 2. Extraer resultados
            df_results = batch_result['dataframe']
            summary = batch_result['summary']

            # 3. Generar análisis
            genai_analysis = self.interpretation_service.interpret_batch_predictions(
                summary
            )

            # 4. Formatear resumen
            summary_text = self._format_batch_summary(summary, genai_analysis)

            logger.info(f"UI batch completed")

            return df_results, summary_text

        except Exception as e:
            logger.error(f"UI batch error: {e}")
            error_msg = f"""
# Error al procesar el lote

**Error:** {str(e)}

**Solución:**
- Verifica que el CSV tenga el formato correcto (separado por ;)
- Debe contener las 11 columnas requeridas
- Ejemplo: fixed acidity;volatile acidity;citric acid;...
            """
            return pd.DataFrame(), error_msg

    def _load_example_handler(self) -> list:
        """
        Carga valores de ejemplo

        Returns:
            Lista con valores de ejemplo para los sliders
        """
        example = create_sample_wine()

        # Retornar en el orden de los inputs
        return [
            example['fixed acidity'],
            example['volatile acidity'],
            example['citric acid'],
            example['residual sugar'],
            example['chlorides'],
            example['free sulfur dioxide'],
            example['total sulfur dioxide'],
            example['density'],
            example['pH'],
            example['sulphates'],
            example['alcohol']
        ]

    def _format_batch_summary(self, summary, genai_analysis: str) -> str:
        """Formatea el resumen del lote"""
        total = summary['total_samples']
        high_count = summary['high_quality_count']
        low_count = summary['low_quality_count']
        high_pct = summary['high_quality_percentage']
        low_pct = summary['low_quality_percentage']

        model_info = summary.get('model_info', {})

        return f"""
# 📊 RESUMEN DEL ANÁLISIS

## Estadísticas Generales
- **Total de vinos:** {total}
- **Alta calidad (≥6):** {high_count} ({high_pct:.1f}%)
- **Baja calidad (<6):** {low_count} ({low_pct:.1f}%)

## 🤖 Análisis del Sommelier AI

{genai_analysis}

---

### Información del Modelo
- **Modelo:** {model_info.get('name', 'wine-quality-classifier')}
- **Versión:** {model_info.get('version', 'N/A')}
- **Stage:** {model_info.get('stage', 'Production')}

💡 **Tip:** Descarga los resultados usando el botón de descarga en la tabla.
        """

    def create_interface(self) -> gr.Blocks:
        """
        Crea la interfaz completa de Gradio

        Returns:
            Interfaz de Gradio configurada
        """
        logger.info("Creating Gradio interface...")

        # Obtener metadata del modelo
        model_metadata = self.prediction_service.get_model_metadata()

        # Tema
        theme = gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        )

        with gr.Blocks(title="🍷 Wine Quality Predictor", theme=theme) as interface:
            # Header
            gr.Markdown("""
            # 🍷 Wine Quality Predictor
            ### Predicción con ML + Interpretación con IA Generativa
            """)

            # Info del sistema
            with gr.Accordion("ℹ️ Información del Sistema", open=False):
                gr.Markdown(f"""
                **Modelo:** {model_metadata.get('model_name', 'N/A')}  
                **Versión:** {model_metadata.get('version', 'N/A')}  
                **Stage:** {model_metadata.get('stage', 'N/A')}  

                **Clasificación:**
                - Alta Calidad: score ≥ 6
                - Baja Calidad: score < 6
                """)

            gr.Markdown("---")

            # Tabs principales
            with gr.Tabs():
                # TAB 1: Predicción Individual
                with gr.Tab("🔍 Análisis Individual"):
                    gr.Markdown("### Ingresa las características del vino")

                    load_example_btn = gr.Button("📝 Cargar Ejemplo", size="sm")

                    # Crear sliders
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Acidez y pH**")
                            fixed_acidity = gr.Slider(
                                **self._create_slider_config('fixed acidity')
                            )
                            volatile_acidity = gr.Slider(
                                **self._create_slider_config('volatile acidity')
                            )
                            citric_acid = gr.Slider(
                                **self._create_slider_config('citric acid')
                            )
                            pH = gr.Slider(**self._create_slider_config('pH'))

                        with gr.Column():
                            gr.Markdown("**Azúcares y Densidad**")
                            residual_sugar = gr.Slider(
                                **self._create_slider_config('residual sugar')
                            )
                            density = gr.Slider(**self._create_slider_config('density'))
                            alcohol = gr.Slider(**self._create_slider_config('alcohol'))

                        with gr.Column():
                            gr.Markdown("**Sales y Conservantes**")
                            chlorides = gr.Slider(**self._create_slider_config('chlorides'))
                            free_sulfur_dioxide = gr.Slider(
                                **self._create_slider_config('free sulfur dioxide')
                            )
                            total_sulfur_dioxide = gr.Slider(
                                **self._create_slider_config('total sulfur dioxide')
                            )
                            sulphates = gr.Slider(**self._create_slider_config('sulphates'))

                    gr.Markdown("---")

                    predict_btn = gr.Button("🔮 Predecir Calidad", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### 📊 Resultados")

                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_output = gr.Markdown()
                            model_info_output = gr.Markdown()

                        with gr.Column(scale=2):
                            interpretation_output = gr.Textbox(
                                label="🤖 Explicación del Sommelier AI",
                                lines=7,
                                show_copy_button=True,
                                interactive=False
                            )

                    # Conectar eventos
                    input_components = [
                        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol
                    ]

                    predict_btn.click(
                        fn=self._predict_single_handler,
                        inputs=input_components,
                        outputs=[prediction_output, interpretation_output, model_info_output]
                    )

                    load_example_btn.click(
                        fn=self._load_example_handler,
                        inputs=None,
                        outputs=input_components
                    )

                # TAB 2: Predicción por Lote
                with gr.Tab("📊 Análisis por Lote (CSV)"):
                    gr.Markdown("""
                    ### Análisis masivo de múltiples vinos

                    Sube un CSV (separado por `;`) con las 11 columnas requeridas.
                    """)

                    file_input = gr.File(label="📁 Subir CSV", file_types=[".csv"])
                    analyze_btn = gr.Button("📈 Analizar Lote", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### 📊 Resultados")

                    summary_output = gr.Markdown()
                    dataframe_output = gr.Dataframe(
                        label="Resultados Detallados",
                        wrap=True,
                        interactive=False
                    )

                    # Conectar eventos
                    analyze_btn.click(
                        fn=self._predict_batch_handler,
                        inputs=file_input,
                        outputs=[dataframe_output, summary_output]
                    )

                # TAB 3: Información
                with gr.Tab("ℹ️ Ayuda"):
                    gr.Markdown("""
                    # 📚 Guía de Uso
                    
                    ## 🔍 Análisis Individual
                    1. Ajusta los valores con los controles deslizantes
                    2. O haz clic en "Cargar Ejemplo"
                    3. Presiona "Predecir Calidad"
                    4. Obtén predicción + explicación AI
                    
                    ## 📊 Análisis por Lote
                    1. Prepara un CSV separado por `;`
                    2. Debe tener las 11 columnas requeridas
                    3. Sube el archivo
                    4. Obtén tabla de resultados + análisis general
                    
                    ## 📖 Columnas Requeridas
```
                    fixed acidity;volatile acidity;citric acid;residual sugar;
                    chlorides;free sulfur dioxide;total sulfur dioxide;
                    density;pH;sulphates;alcohol
```
                    
                    ## 🤖 Tecnologías
                    - **MLflow:** Gestión de modelos
                    - **Random Forest:** Algoritmo de clasificación
                    - **Gemini AI:** Explicaciones generativas
                    - **Gradio:** Interfaz de usuario
                    
                    ---
                    
                    Para más información, consulta la documentación del proyecto.
                    """)

            # Footer
            gr.Markdown("---")
            gr.Markdown("""
            <div style='text-align: center; color: #666;'>
                <p>🔬 MLflow • Random Forest • Gemini AI • Gradio</p>
            </div>
            """)

        logger.info("Gradio interface created")

        return interface

    def _create_slider_config(self, feature_name: str) -> dict:
        """
        Crea configuración para un slider de Gradio

        Args:
            feature_name: Nombre de la feature

        Returns:
            Diccionario con configuración del slider
        """
        ranges = self.feature_ranges[feature_name]

        # Determinar step según feature
        if 'chlorides' in feature_name or 'sulphates' in feature_name:
            step = 0.001
        elif 'acidity' in feature_name or 'citric' in feature_name:
            step = 0.01
        elif 'density' in feature_name:
            step = 0.0001
        elif 'sulfur' in feature_name:
            step = 1.0
        else:
            step = 0.1

        # Etiquetas descriptivas
        labels = {
            'fixed acidity': "Acidez Fija (g/dm³)",
            'volatile acidity': "Acidez Volátil (g/dm³)",
            'citric acid': "Ácido Cítrico (g/dm³)",
            'residual sugar': "Azúcar Residual (g/dm³)",
            'chlorides': "Cloruros (g/dm³)",
            'free sulfur dioxide': "SO₂ Libre (mg/dm³)",
            'total sulfur dioxide': "SO₂ Total (mg/dm³)",
            'density': "Densidad (g/cm³)",
            'pH': "pH",
            'sulphates': "Sulfatos (g/dm³)",
            'alcohol': "Alcohol (% vol.)"
        }

        return {
            'minimum': ranges['min'],
            'maximum': ranges['max'],
            'value': ranges['typical'],
            'step': step,
            'label': labels.get(feature_name, feature_name)
        }

    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False
    ):
        """
        Crea y lanza la interfaz de Gradio

        Args:
            server_name: Host del servidor
            server_port: Puerto del servidor
            share: Si crear link público
            debug: Modo debug
        """
        logger.info("Launching Gradio interface...")

        interface = self.create_interface()

        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True
        )