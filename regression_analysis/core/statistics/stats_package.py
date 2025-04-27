"""
Пакет статистических расчетов для регрессионного анализа.

Предоставляет функции для расчета статистических показателей моделей
регрессии, включая R², F-статистику, t-тесты, доверительные интервалы и 
другие метрики, необходимые для оценки качества моделей.
"""

# Импорты из модуля base_statistics
from core.statistics.base_statistics import (
    limited_statistics
)

# Импорты из модуля regression_stats
from core.statistics.regression_stats import (
    calculate_statistics,
    calculate_r_squared,
    calculate_adjusted_r_squared,
    calculate_standard_error
)

# Импорты из модуля coefficient_stats
from core.statistics.coefficient_stats import (
    calculate_all_statistics,
    calculate_coefficient_pvalues,
    calculate_confidence_intervals
)

# Определение экспортируемых функций и классов
__all__ = [
    # Из base_statistics
    'limited_statistics',
    
    # Из regression_stats
    'calculate_statistics',
    'calculate_r_squared',
    'calculate_adjusted_r_squared',
    'calculate_standard_error',
    
    # Из coefficient_stats
    'calculate_all_statistics',
    'calculate_coefficient_pvalues',
    'calculate_confidence_intervals'
]