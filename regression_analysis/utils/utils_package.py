"""
Пакет вспомогательных утилит для приложения регрессионного анализа.

Содержит модули для визуализации данных, экспорта результатов и 
других вспомогательных функций, используемых в приложении.
"""

# Импорты из подпакета visualization
from utils.visualization.graph_manager import (
    create_graph,
    export_all_plots
)
from utils.visualization.basic_plots import (
    create_actual_vs_predicted_plot,
    create_residuals_plot
)
from utils.visualization.coefficient_plots import (
    create_coefficient_bar_plot,
    create_importance_plot
)
from utils.visualization.residual_plots import (
    create_residual_histogram,
    create_qq_plot
)

# Импорты из подпакета export
from utils.export.pdf_exporter import (
    export_to_pdf,
    create_pdf_report
)
from utils.export.csv_exporter import (
    export_to_csv,
    export_coefficients_to_csv,
    export_predictions_to_csv
)

# Определение экспортируемых функций и классов
__all__ = [
    # Из visualization/graph_manager
    'create_graph',
    'export_all_plots',
    
    # Из visualization/basic_plots
    'create_actual_vs_predicted_plot',
    'create_residuals_plot',
    
    # Из visualization/coefficient_plots
    'create_coefficient_bar_plot',
    'create_importance_plot',
    
    # Из visualization/residual_plots
    'create_residual_histogram',
    'create_qq_plot',
    
    # Из export/pdf_exporter
    'export_to_pdf',
    'create_pdf_report',
    
    # Из export/csv_exporter
    'export_to_csv',
    'export_coefficients_to_csv',
    'export_predictions_to_csv'
]