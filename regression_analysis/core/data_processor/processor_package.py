"""
Пакет для обработки данных в приложении регрессионного анализа.

Предоставляет функции для загрузки, очистки и подготовки данных 
для регрессионного анализа, а также выбора признаков.
"""

# Импорты из модуля data_loader
from core.data_processor.data_loader import (
    prepare_data,
    read_csv_file,
    detect_encoding_and_delimiter
)

# Импорты из модуля data_cleaner
from core.data_processor.data_cleaner import (
    clean_data,
    handle_missing_values,
    convert_to_numeric
)

# Импорты из модуля feature_selector
from core.data_processor.feature_selector import (
    select_features,
    detect_age_groups,
    find_correlated_features
)

# Определение экспортируемых функций и классов
__all__ = [
    # Из data_loader
    'prepare_data',
    'read_csv_file',
    'detect_encoding_and_delimiter',
    
    # Из data_cleaner
    'clean_data',
    'handle_missing_values',
    'convert_to_numeric',
    
    # Из feature_selector
    'select_features',
    'detect_age_groups',
    'find_correlated_features'
]