"""
Пакет пользовательского интерфейса для приложения регрессионного анализа.

Содержит классы и функции для создания и управления элементами пользовательского
интерфейса, включая диалоги, окна просмотра и компоненты отображения.
"""

# Импорты из компонентов пользовательского интерфейса
from ui.components.ui_helpers import center_window

# Импорты из диалоговых окон
from ui.dialogs.csv_dialog import CSVSettingsDialog
from ui.dialogs.file_selector import FileSelector

# Импорты из просмотрщиков
from ui.viewers.coefficient_viewer import CoefficientViewer
from ui.viewers.graph_viewer import GraphViewer
from ui.viewers.statistics_viewer import StatisticsViewer

# Импорты основных элементов UI
from ui.regression_app import RegressionApp
from ui.model_display import ModelDisplayFrame

# Определение экспортируемых классов и функций
__all__ = [
    # Вспомогательные функции интерфейса
    'center_window',
    
    # Диалоговые окна
    'CSVSettingsDialog',
    'FileSelector',
    
    # Просмотрщики
    'CoefficientViewer',
    'GraphViewer',
    'StatisticsViewer',
    
    # Основные элементы интерфейса
    'RegressionApp',
    'ModelDisplayFrame'
]