"""
Пакет диалоговых окон для приложения регрессионного анализа.

Содержит классы для создания и управления различными диалоговыми окнами,
такими как настройка параметров CSV, выбор файлов, настройка моделей и т.д.
"""

# Импорты из модуля csv_dialog
from ui.dialogs.csv_dialog import (
    CSVSettingsDialog
)

# Импорты из модуля file_selector 
from ui.dialogs.file_selector import (
    FileSelector,
    FileOpenDialog,
    FileSaveDialog
)

# В будущем могут быть добавлены другие диалоги:
# - ModelConfigDialog
# - ExportSettingsDialog
# - PreferencesDialog
# - HelpDialog

# Определение экспортируемых классов
__all__ = [
    # Из csv_dialog
    'CSVSettingsDialog',
    
    # Из file_selector
    'FileSelector',
    'FileOpenDialog',
    'FileSaveDialog'
    
    # Здесь можно будет добавить дополнительные диалоги по мере их создания
]