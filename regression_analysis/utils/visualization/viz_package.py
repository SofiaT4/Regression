"""
Пакет для визуализации данных и результатов регрессионного анализа.

Содержит модули и функции для создания графиков, диаграмм и других
визуальных представлений данных, используемых в приложении.
"""

# Импорты из модуля graph_manager
from utils.visualization.graph_manager import (
    create_graph,
    export_all_plots
)

# Импорты из модуля basic_plots
from utils.visualization.basic_plots import (
    create_actual_vs_predicted_plot,
    create_scatter_plot,
    create_line_plot,
    create_bar_plot
)

# Импорты из модуля coefficient_plots
from utils.visualization.coefficient_plots import (
    create_coefficient_bar_plot,
    create_importance_plot,
    create_significance_plot,
    create_correlation_matrix
)

# Импорты из модуля residual_plots
from utils.visualization.residual_plots import (
    create_residual_scatter_plot,
    create_residual_histogram,
    create_qq_plot,
    create_standardized_residual_plot
)

# Определение экспортируемых функций и классов
__all__ = [
    # Из graph_manager
    'create_graph',
    'export_all_plots',
    
    # Из basic_plots
    'create_actual_vs_predicted_plot',
    'create_scatter_plot',
    'create_line_plot',
    'create_bar_plot',
    
    # Из coefficient_plots
    'create_coefficient_bar_plot',
    'create_importance_plot',
    'create_significance_plot',
    'create_correlation_matrix',
    
    # Из residual_plots
    'create_residual_scatter_plot',
    'create_residual_histogram',
    'create_qq_plot',
    'create_standardized_residual_plot'
]