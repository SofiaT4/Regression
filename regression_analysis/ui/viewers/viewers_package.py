"""
Пакет модулей просмотра данных для приложения регрессионного анализа.

Содержит классы для визуализации и представления результатов анализа,
включая коэффициенты регрессии, графики и статистические показатели.
"""

# Импорты из модуля coefficient_viewer
from ui.viewers.coefficient_viewer import (
    CoefficientViewer
)

# Импорты из модуля graph_viewer
from ui.viewers.graph_viewer import (
    GraphViewer
)

# Импорты из модуля statistics_viewer
from ui.viewers.statistics_viewer import (
    StatisticsViewer
)

# В будущем могут быть добавлены другие компоненты просмотра:
# - RegressionResultViewer
# - DataTableViewer
# - ResidualAnalysisViewer
# - ScatterPlotViewer

# Определение экспортируемых классов
__all__ = [
    # Из coefficient_viewer
    'CoefficientViewer',
    
    # Из graph_viewer
    'GraphViewer',
    
    # Из statistics_viewer
    'StatisticsViewer'
    
    # Здесь можно будет добавить дополнительные классы по мере их создания
]