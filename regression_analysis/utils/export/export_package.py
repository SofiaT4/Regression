"""
Пакет для экспорта данных и результатов регрессионного анализа.

Содержит модули и функции для сохранения данных, результатов анализа,
графиков и отчетов в различных форматах (PDF, CSV, Excel и т.д.).
"""

# Импорты из модуля pdf_exporter
from utils.export.pdf_exporter import (
    export_to_pdf,
    create_pdf_report,
    export_figure_to_pdf,
    export_multiple_figures
)

# Импорты из модуля csv_exporter
from utils.export.csv_exporter import (
    export_to_csv,
    export_coefficients_to_csv,
    export_predictions_to_csv,
    export_statistics_to_csv
)

# В будущем могут быть добавлены другие модули экспорта:
# - excel_exporter
# - json_exporter
# - report_generator

# Определение экспортируемых функций и классов
__all__ = [
    # Из pdf_exporter
    'export_to_pdf',
    'create_pdf_report',
    'export_figure_to_pdf',
    'export_multiple_figures',
    
    # Из csv_exporter
    'export_to_csv',
    'export_coefficients_to_csv',
    'export_predictions_to_csv',
    'export_statistics_to_csv'
    
    # Здесь можно будет добавить дополнительные функции по мере их создания
]