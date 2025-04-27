"""
Пакет для работы с регрессионными моделями в приложении.

Предоставляет функции для построения моделей линейной регрессии 
и форматирования результатов моделей для отображения.
"""

# Импорты из модуля regression_builder
from core.models.regression_builder import (
    build_models,
    build_regression_model,
    predict_with_model,
    handle_model_errors
)

# Импорты из модуля model_formatter
from core.models.model_formatter import (
    format_equation_for_display,
    format_equation_for_charts,
    simplify_feature_name,
    format_coefficient_table
)

# Определение экспортируемых функций и классов
__all__ = [
    # Из regression_builder
    'build_models',
    'build_regression_model',
    'predict_with_model',
    'handle_model_errors',
    
    # Из model_formatter
    'format_equation_for_display',
    'format_equation_for_charts',
    'simplify_feature_name',
    'format_coefficient_table'
]