"""
Utilidades para formateo de datos
"""

from typing import Dict, List, Optional

# Mapeo de nombres técnicos a descriptivos
FEATURE_DESCRIPTIONS = {
    'fixed acidity': {
        'name': 'Acidez fija',
        'unit': 'g/dm³',
        'description': 'Ácidos tartárico y málico'
    },
    'volatile acidity': {
        'name': 'Acidez volátil',
        'unit': 'g/dm³',
        'description': 'Ácido acético'
    },
    'citric acid': {
        'name': 'Ácido cítrico',
        'unit': 'g/dm³',
        'description': 'Añade frescura'
    },
    'residual sugar': {
        'name': 'Azúcar residual',
        'unit': 'g/dm³',
        'description': 'Dulzor del vino'
    },
    'chlorides': {
        'name': 'Cloruros',
        'unit': 'g/dm³',
        'description': 'Salinidad'
    },
    'free sulfur dioxide': {
        'name': 'SO₂ libre',
        'unit': 'mg/dm³',
        'description': 'Conservante activo'
    },
    'total sulfur dioxide': {
        'name': 'SO₂ total',
        'unit': 'mg/dm³',
        'description': 'Conservante total'
    },
    'density': {
        'name': 'Densidad',
        'unit': 'g/cm³',
        'description': 'Depende de alcohol y azúcar'
    },
    'pH': {
        'name': 'pH',
        'unit': '',
        'description': 'Nivel de acidez'
    },
    'sulphates': {
        'name': 'Sulfatos',
        'unit': 'g/dm³',
        'description': 'Conservante adicional'
    },
    'alcohol': {
        'name': 'Alcohol',
        'unit': '% vol.',
        'description': 'Contenido alcohólico'
    }
}


def format_feature_value(feature_name: str, value: float) -> str:
    """
    Formatea un valor de feature con su unidad

    Args:
        feature_name: Nombre de la feature
        value: Valor numérico

    Returns:
        String formateado con valor y unidad
    """
    if feature_name not in FEATURE_DESCRIPTIONS:
        return f"{value}"

    info = FEATURE_DESCRIPTIONS[feature_name]
    unit = info['unit']

    # Determinar formato según feature
    if 'acidity' in feature_name or 'citric' in feature_name:
        formatted = f"{value:.2f}"
    elif 'chlorides' in feature_name or 'sulphates' in feature_name:
        formatted = f"{value:.3f}"
    elif 'sulfur' in feature_name:
        formatted = f"{value:.0f}"
    elif 'density' in feature_name:
        formatted = f"{value:.4f}"
    elif 'pH' in feature_name or 'alcohol' in feature_name:
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:.1f}"

    if unit:
        return f"{formatted} {unit}"
    return formatted


def format_features_for_display(
        features: Dict[str, float],
        highlight_features: Optional[List[str]] = None
) -> str:
    """
    Formatea features para mostrar en texto

    Args:
        features: Diccionario con features
        highlight_features: Features a destacar (opcional)

    Returns:
        String multilínea formateado
    """
    lines = []

    for feature_name, value in features.items():
        if feature_name not in FEATURE_DESCRIPTIONS:
            continue

        info = FEATURE_DESCRIPTIONS[feature_name]
        display_name = info['name']
        description = info['description']
        formatted_value = format_feature_value(feature_name, value)

        # Destacar si es feature importante
        prefix = "⭐ " if (highlight_features and feature_name in highlight_features) else "- "

        line = f"{prefix}**{display_name}:** {formatted_value} ({description})"
        lines.append(line)

    return "\n".join(lines)


def format_prediction_label(prediction: int, include_emoji: bool = True) -> str:
    """
    Formatea etiqueta de predicción

    Args:
        prediction: 0 o 1
        include_emoji: Si incluir emoji

    Returns:
        String formateado
    """
    if prediction == 1:
        label = "Alta Calidad (≥6)"
        emoji = "🍷✨" if include_emoji else ""
    else:
        label = "Baja Calidad (<6)"
        emoji = "🍷" if include_emoji else ""

    return f"{emoji} {label}".strip()


def format_confidence(confidence: float) -> str:
    """
    Formatea valor de confianza

    Args:
        confidence: Valor entre 0 y 1

    Returns:
        String formateado con porcentaje
    """
    percentage = confidence * 100
    return f"{percentage:.1f}%"