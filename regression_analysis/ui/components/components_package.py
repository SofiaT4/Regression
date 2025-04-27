"""
Пакет компонентов пользовательского интерфейса для приложения регрессионного анализа.

Содержит вспомогательные функции и базовые компоненты интерфейса, 
которые используются в различных частях приложения.
"""

# Импорты вспомогательных функций интерфейса
from ui.components.ui_helpers import (
    center_window,
    create_scrollable_frame,
    create_labeled_entry,
    create_tooltip,
    validate_numeric_input
)

# В дальнейшем здесь могут быть импорты дополнительных компонентов, таких как:
# - CustomButton
# - StatusBar
# - ThemeManager
# - FormattedLabel
# и т.д.

# Определение экспортируемых функций и классов
__all__ = [
    # Вспомогательные функции интерфейса
    'center_window',
    'create_scrollable_frame',
    'create_labeled_entry',
    'create_tooltip',
    'validate_numeric_input'
    
    # Здесь можно будет добавить дополнительные компоненты по мере их создания
]