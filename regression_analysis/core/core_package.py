"""
Центральный пакет для бизнес-логики приложения регрессионного анализа.

Содержит подмодули для обработки данных, построения регрессионных моделей 
и расчета статистических показателей.
"""

# Импорт подмодулей data_processor
from core.data_processor.data_loader import prepare_data
from core.data_processor.data_cleaner import clean_data
from core.data_processor.feature_selector import select_features

# Импорт подмодулей models
from core.models.regression_builder import build_models
from core.models.model_formatter import (
    format_equation_for_display,
    format_equation_for_charts,
    simplify_feature_name
)

# Импорт подмодулей statistics
from core.statistics.base_statistics import limited_statistics
from core.statistics.regression_stats import calculate_statistics
from core.statistics.coefficient_stats import calculate_all_statistics

# Экспортируемые функции и классы
__all__ = [
    # Из data_processor
    'prepare_data',
    'clean_data',
    'select_features',
    
    # Из models
    'build_models',
    'format_equation_for_display',
    'format_equation_for_charts',
    'simplify_feature_name',
    
    # Из statistics
    'limited_statistics',
    'calculate_statistics',
    'calculate_all_statistics'
]