"""
Вспомогательные функции для пользовательского интерфейса.
Содержит набор функций для создания и настройки элементов UI, обработки событий и форматирования данных.
"""

import tkinter as tk
from tkinter import ttk
import re
from typing import Any, Optional, Tuple, Dict, List, Callable, Union

# Импортируем функции стилизации из модуля theme_manager
from ui.components.theme_manager import apply_theme, DARK_THEME, style_treeview_tags

def center_window(window: Union[tk.Tk, tk.Toplevel], width: int, height: int) -> None:
    """Центрирует окно на экране."""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def apply_style(root: Union[tk.Tk, tk.Toplevel], style_name: str = "dark") -> None:
    """
    Применяет указанный стиль к приложению.
    
    Parameters:
    root (tk.Tk or tk.Toplevel): Корневое окно приложения
    style_name (str): Название стиля ('dark', 'light', 'system', etc.)
    """
    if style_name.lower() == "dark":
        apply_theme(root)
    else:
        # Для других стилей можно добавить аналогичные функции
        pass

def create_scrollable_frame(parent: tk.Widget, **kwargs) -> Tuple[ttk.Frame, ttk.Frame]:
    """Создает фрейм с полосами прокрутки."""
    container = ttk.Frame(parent)
    container.pack(fill=tk.BOTH, expand=True, **kwargs)
    
    canvas = tk.Canvas(container, bd=0, highlightthickness=0, 
                     bg=DARK_THEME['primary'])
    scrollbar_y = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
    
    def update_scrollregion(event: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    scrollable_frame.bind("<Configure>", update_scrollregion)
    
    def update_frame_width(event: tk.Event) -> None:
        canvas.itemconfig(scrollable_frame_id, width=event.width)
    
    canvas.bind("<Configure>", update_frame_width)
    
    # Привязываем события колеса мыши для прокрутки
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    
    return container, scrollable_frame

def create_labeled_entry(
    parent: tk.Widget, 
    text: str, 
    variable: Optional[tk.Variable] = None, 
    width: int = 20, 
    validate_func: Optional[Callable[[str], bool]] = None,
    **kwargs
) -> Tuple[tk.Frame, tk.Label, tk.Entry]:
    """Создает фрейм с меткой и полем ввода."""
    frame = tk.Frame(parent, bg=DARK_THEME['primary'])
    label = tk.Label(frame, text=text, anchor="w", bg=DARK_THEME['primary'], 
                   fg=DARK_THEME['neutral'])
    label.pack(side=tk.LEFT, padx=5)
    
    if validate_func:
        validate_command = parent.register(validate_func)
        entry = tk.Entry(
            frame, 
            textvariable=variable, 
            width=width, 
            validate="key", 
            validatecommand=(validate_command, '%P'),
            bg=DARK_THEME['bg'],
            fg=DARK_THEME['text_light'],
            insertbackground=DARK_THEME['accent'],  # цвет курсора
            **kwargs
        )
    else:
        entry = tk.Entry(
            frame, 
            textvariable=variable, 
            width=width,
            bg=DARK_THEME['bg'],
            fg=DARK_THEME['text_light'],
            insertbackground=DARK_THEME['accent'],
            **kwargs
        )
    
    entry.pack(side=tk.LEFT, padx=5)
    
    return frame, label, entry

def create_tooltip(widget: tk.Widget, text: str) -> None:
    """Создает всплывающую подсказку для виджета."""
    tooltip = None
    
    def enter(event: tk.Event) -> None:
        nonlocal tooltip
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = tk.Frame(tooltip, bd=1, bg=DARK_THEME['bg_light'])
        frame.pack(ipadx=5, ipady=5)
        
        label = tk.Label(frame, text=text, bg=DARK_THEME['bg_light'], 
                      fg=DARK_THEME['text_light'], justify=tk.LEFT, wraplength=250)
        label.pack()
    
    def leave(event: tk.Event) -> None:
        nonlocal tooltip
        if tooltip:
            tooltip.destroy()
            tooltip = None
    
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

def validate_numeric_input(value: str, allow_float: bool = True) -> bool:
    """Проверяет, является ли введенное значение числом."""
    if value == "":
        return True
    
    if allow_float:
        pattern = r'^[+-]?(\d+([.,]\d*)?|[.,]\d+)$'
    else:
        pattern = r'^[+-]?\d+$'
    
    return bool(re.match(pattern, value))

def create_button(
    parent: tk.Widget, 
    text: str, 
    command: Callable, 
    style: str = "default", 
    **kwargs
) -> ttk.Button:
    """
    Создает стилизованную кнопку.
    
    Parameters:
    parent (tk.Widget): Родительский виджет
    text (str): Текст кнопки
    command (Callable): Функция, вызываемая при нажатии
    style (str): Стиль кнопки ('default', 'accent', 'secondary')
    **kwargs: Дополнительные параметры
    
    Returns:
    ttk.Button: Созданная кнопка
    """
    if style == "accent":
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton", **kwargs)
    elif style == "secondary":
        button = ttk.Button(parent, text=text, command=command, style="Secondary.TButton", **kwargs)
    else:
        button = ttk.Button(parent, text=text, command=command, **kwargs)
    
    return button