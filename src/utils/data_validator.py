"""
Utilidades para validación de datos
"""

from typing import Dict, List, Set, Union
import pandas as pd

# Definición de features esperadas
WINE_FEATURES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]


def get_feature_names() -> List[str]:
    return WINE_FEATURES.copy()


def validate_features(data: Union[Dict, pd.DataFrame]) -> Dict[str, Union[bool, List[str]]]:
    """
    Valida que los datos tengan las features correctas

    Args:
        data: Diccionario o DataFrame con datos

    Returns:
        Diccionario con resultado de validación:
        {
            'is_valid': bool,
            'missing_features': List[str],
            'extra_features': List[str],
            'expected_features': List[str]
        }
    """
    # Obtener columnas del input
    if isinstance(data, dict):
        provided_features = set(data.keys())
    elif isinstance(data, pd.DataFrame):
        provided_features = set(data.columns)
    else:
        raise TypeError("Data must be dict or DataFrame")

    expected_features = set(WINE_FEATURES)

    missing = list(expected_features - provided_features)
    extra = list(provided_features - expected_features)

    return {
        'is_valid': len(missing) == 0,
        'missing_features': missing,
        'extra_features': extra,
        'expected_features': WINE_FEATURES
    }


def prepare_dataframe(data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepara DataFrame asegurando orden correcto de columnas

    Args:
        data: Dict o DataFrame con datos

    Returns:
        DataFrame con columnas ordenadas correctamente

    Raises:
        ValueError: Si faltan features requeridas
    """
    # Convertir a DataFrame si es dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
        data_float = {k: float(v) for k, v in data.items()}
        df = pd.DataFrame([data_float])
    else:
        df = data.copy()

    # Validar
    validation = validate_features(df)


    if not validation['is_valid']:
        missing = validation['missing_features']
        raise ValueError(f"Missing required features: {missing}")

    # Seleccionar y ordenar columnas
    df = df[WINE_FEATURES]
    df = df.astype('float64')
    return df


def get_feature_ranges() -> Dict[str, Dict[str, float]]:
    """
    Retorna rangos típicos de cada feature

    Returns:
        Diccionario con min, max y valor típico de cada feature
    """
    return {
        'fixed acidity': {'min': 4.0, 'max': 16.0, 'typical': 7.0},
        'volatile acidity': {'min': 0.1, 'max': 1.6, 'typical': 0.3},
        'citric acid': {'min': 0.0, 'max': 1.0, 'typical': 0.3},
        'residual sugar': {'min': 0.6, 'max': 66.0, 'typical': 5.0},
        'chlorides': {'min': 0.01, 'max': 0.35, 'typical': 0.05},
        'free sulfur dioxide': {'min': 2.0, 'max': 290.0, 'typical': 30.0},
        'total sulfur dioxide': {'min': 9.0, 'max': 440.0, 'typical': 120.0},
        'density': {'min': 0.98, 'max': 1.04, 'typical': 0.995},
        'pH': {'min': 2.7, 'max': 4.0, 'typical': 3.2},
        'sulphates': {'min': 0.22, 'max': 1.1, 'typical': 0.5},
        'alcohol': {'min': 8.0, 'max': 14.9, 'typical': 10.5}
    }


def create_sample_wine() -> Dict[str, float]:
    """
    Crea un vino de ejemplo con valores típicos

    Returns:
        Diccionario con features de ejemplo
    """
    ranges = get_feature_ranges()
    return {
        feature: info['typical']
        for feature, info in ranges.items()
    }