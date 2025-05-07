"""
Вспомогательные функции для пользовательского интерфейса.
Содержит набор функций для создания и настройки элементов UI, обработки событий и форматирования данных.
"""

import tkinter as tk
from tkinter import ttk
import re
from typing import Any, Optional, Tuple, Dict, List, Callable, Union
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import os
import matplotlib

# Импортируем функции стилизации из модуля theme_manager
from ui.components.theme_manager import apply_theme, DARK_THEME, style_treeview_tags

# Словарь с русскими переводами для всплывающих подсказок NavigationToolbar2Tk
TOOLBAR_TOOLTIPS = {
    'Home': 'Домой',
    'Back': 'Назад',
    'Forward': 'Вперед',
    'Pan': 'Панорамирование',
    'Zoom': 'Масштабирование',
    'Zoom to rectangle': 'Масштабирование области',
    'Subplots': 'Настройка графика',
    'Save': 'Сохранить фигуру',
    'help': 'Помощь',
    'quit': 'Выход'
}

# Словарь для прямого перевода текстов кнопок
BUTTON_TEXTS = {
    'Zoom to rectangle': 'Масштабирование области',
    'Pan Axes': 'Панорамирование',
    'x/y fixes axis': 'Фиксация осей x/y',
    'zoom rect': 'Масштабирование',
    'pan/zoom': 'Панорамирование',
    'Zoom': 'Масштабирование',
    'Pan': 'Панорамирование',
    'Home': 'Домой',
    'Back': 'Назад',
    'Forward': 'Вперед',
    'Subplots': 'Настройка графика',
    'Save': 'Сохранить'
}

class RussianNavigationToolbar(NavigationToolbar2Tk):
    """
    Панель навигации для графиков matplotlib с русскими подсказками.
    Расширяет стандартный NavigationToolbar2Tk для обеспечения полной 
    функциональности всех элементов управления и русификации подсказок.
    """
    # Переопределяем элементы тулбара с русскими названиями
    toolitems = (
        ('Домой', 'Восстановить исходный вид', 'home', 'home'),
        ('Назад', 'Вернуться к предыдущему виду', 'back', 'back'),
        ('Вперед', 'Перейти к следующему виду', 'forward', 'forward'),
        ('Панорамирование', 'Панорамирование осей', 'move', 'pan'),
        ('Масштабирование', 'Масштабирование области', 'zoom_to_rect', 'zoom'),
        ('Настройка графика', 'Настроить график', 'subplots', 'configure_subplots'),
        ('Сохранить', 'Сохранить фигуру', 'filesave', 'save_figure'),
    )
    
    # Переводы для сообщений в панели инструментов
    message_translations = {
        'Zoom to rectangle': 'Масштабирование области',
        'Zoom rectangle': 'Область масштабирования',
        'Pan': 'Панорамирование',
        'Pan axes': 'Панорамирование осей',
        'Navigate mode': 'Режим навигации',
        'Press left button to pan, right button to zoom': 'Нажмите левую кнопку для панорамирования, правую для масштабирования',
        'Press CTRL and move the mouse to zoom': 'Нажмите CTRL и двигайте мышь для масштабирования',
        'Use wheel to zoom': 'Используйте колесо для масштабирования',
        'Home/Reset': 'Домой/Сброс',
        'Reset original view': 'Восстановить исходный вид',
        'Back': 'Назад',
        'Previous view': 'Предыдущий вид',
        'Forward': 'Вперед',
        'Next view': 'Следующий вид',
        'Save': 'Сохранить',
        'Save the figure': 'Сохранить фигуру',
        'Subplots': 'Настройка графика',
        'Configure subplots': 'Настроить график',
        'x/y fixes axis': 'Фиксация осей x/y'
    }
    
    def __init__(self, canvas, parent, theme=None):
        # Инициализируем родительский класс с нашими русскими элементами (toolitems)
        NavigationToolbar2Tk.__init__(self, canvas, parent)
        self.theme = theme
        
        # Словарь для доступа к кнопкам
        self._buttons = {}
        
        # Находим кнопки и сохраняем их для быстрого доступа
        for i, child in enumerate(self.winfo_children()):
            if isinstance(child, tk.Button):
                # Определяем тип кнопки по индексу в наборе
                if i < len(self.toolitems):
                    name, tooltip, _, callback = self.toolitems[i]
                    self._buttons[name] = child
                    # Создаем подсказку на русском
                    create_tooltip(child, tooltip)
        
        # Принудительно обновляем состояние кнопок
        self._update_buttons_checked()
        
        # Применяем темную тему, если она предоставлена
        if theme:
            self.config(background=theme['primary'])
            for button in self.winfo_children():
                if isinstance(button, tk.Button):
                    button.config(
                        bg=theme['bg_light'],
                        fg=theme['neutral'],
                        activebackground=theme['accent'],
                        activeforeground=theme['text_light']
                    )
                elif isinstance(button, tk.Label):
                    button.config(
                        background=theme['primary'],
                        foreground=theme['neutral']
                    )
                    
    def _make_classic_style_pseudo_toolbar(self):
        """
        Создаем тулбар в классическом стиле с русскими названиями.
        Это переопределенный метод из NavigationToolbar2Tk.
        """
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
        
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.pack(tk.Label(self, text=' '), side=tk.LEFT)
                continue
            
            image = os.path.join(self.basedir, image_file + '.png')
            im = tk.PhotoImage(master=self, file=image)
            button = tk.Button(
                self, text=text, padx=2, pady=2, image=im, command=getattr(self, callback))
            button._ntimage = im
            button.pack(side=tk.LEFT)
            if callback in ['zoom', 'pan']:
                self._active_button = button
            self._buttons[text] = button
    
    def set_message(self, s):
        """
        Переопределяем метод для перевода сообщений на русский.
        """
        if s in self.message_translations:
            s = self.message_translations[s]
        else:
            # Пытаемся искать частичные совпадения для неточных сообщений
            for eng, rus in self.message_translations.items():
                if eng in s:
                    s = s.replace(eng, rus)
                    break
        
        # Вызываем родительский метод с переведенным сообщением
        NavigationToolbar2Tk.set_message(self, s)

    def zoom(self, *args):
        """
        Переопределяем метод зума для обеспечения стабильности работы кнопки.
        """
        NavigationToolbar2Tk.zoom(self, *args)
        # Принудительно устанавливаем режим и обновляем внешний вид
        self.mode = 'zoom rect'
        self.set_message('Масштабирование области')
        # Обновляем состояние кнопок
        self._update_buttons_checked()

    def pan(self, *args):
        """
        Переопределяем метод панорамирования для обеспечения стабильности работы кнопки.
        """
        NavigationToolbar2Tk.pan(self, *args)
        # Принудительно устанавливаем режим и обновляем внешний вид
        self.mode = 'pan/zoom'
        self.set_message('Панорамирование осей')
        # Обновляем состояние кнопок
        self._update_buttons_checked()
        
    # Переопределяем другие методы, чтобы гарантировать их корректную работу с русскими надписями
    def home(self, *args):
        """Сброс к начальному виду."""
        NavigationToolbar2Tk.home(self, *args)
        self.set_message('Восстановить исходный вид')
    
    def back(self, *args):
        """Возврат к предыдущему виду."""
        NavigationToolbar2Tk.back(self, *args)
        self.set_message('Возврат к предыдущему виду')
    
    def forward(self, *args):
        """Переход к следующему виду."""
        NavigationToolbar2Tk.forward(self, *args)
        self.set_message('Переход к следующему виду')
    
    def _update_buttons_checked(self):
        """
        Переопределяем метод обновления состояния кнопок.
        """
        # Вызываем родительский метод для обновления внутреннего состояния
        NavigationToolbar2Tk._update_buttons_checked(self)
        
        # После обновления состояния в родительском классе, принудительно обновляем внешний вид кнопок
        if hasattr(self, '_buttons'):
            # Явно устанавливаем состояние кнопок на основе текущего режима
            if 'Панорамирование' in self._buttons:
                if self.mode == 'pan/zoom':
                    self._buttons['Панорамирование'].config(relief='sunken')
                else:
                    self._buttons['Панорамирование'].config(relief='raised')
                    
            if 'Масштабирование' in self._buttons:
                if self.mode == 'zoom rect':
                    self._buttons['Масштабирование'].config(relief='sunken')
                else:
                    self._buttons['Масштабирование'].config(relief='raised')
    
    def configure_subplots(self, *args):
        """Настройка графика с русификацией окна настройки."""
        # Вызываем метод напрямую из класса, а не через экземпляр
        # Это решит проблему с открытием окна
        result = NavigationToolbar2Tk.configure_subplots(self, *args)
        
        # Если конфигурационное окно было создано
        if hasattr(self, "subplot_tool") and self.subplot_tool:
            # Русифицируем надписи и подсказки в окне настройки
            # Используем title() для Tkinter окна вместо set_title
            self.subplot_tool.title("Настройка подграфиков")

            # Переводим текст инструкции
            for widget in self.subplot_tool.winfo_children():
                if isinstance(widget, tk.Label):
                    if "Click on slider" in widget.cget("text"):
                        widget.config(text="Щелкните на ползунке для настройки параметров графика")

            # Переводим подписи ползунков
            for slider_label in self.subplot_tool.winfo_children():
                if isinstance(slider_label, tk.Label):
                    label_text = slider_label.cget("text")
                    if label_text == "left":
                        slider_label.config(text="левый")
                    elif label_text == "bottom":
                        slider_label.config(text="нижний")
                    elif label_text == "right":
                        slider_label.config(text="правый")
                    elif label_text == "top":
                        slider_label.config(text="верхний")
                    elif label_text == "wspace":
                        slider_label.config(text="гор. отступ")
                    elif label_text == "hspace":
                        slider_label.config(text="верт. отступ")

            # Переводим кнопку "Close"
            for button in self.subplot_tool.winfo_children():
                if isinstance(button, tk.Button) and button.cget("text") == "Close":
                    button.config(text="Закрыть")
                    
            # Принудительное обновление окна
            self.subplot_tool.update()

        self.set_message('Настройка графика')
        return result
    
    def save_figure(self, *args):
        """Сохранение фигуры."""
        NavigationToolbar2Tk.save_figure(self, *args)
        self.set_message('Сохранить фигуру')
    
    def update(self):
        """
        Обновление всех элементов панели инструментов.
        """
        NavigationToolbar2Tk.update(self)
        # Принудительно обновляем состояние кнопок после обновления тулбара
        self._update_buttons_checked()
    
    def press(self, event):
        """
        Обработка нажатия кнопки мыши, обеспечивает корректную работу режимов
        масштабирования и панорамирования.
        """
        # Вызываем родительский метод для обработки нажатия
        NavigationToolbar2Tk.press(self, event)
        
        # Обновляем сообщение в соответствии с текущим режимом
        if self.mode == 'zoom rect':
            self.set_message('Масштабирование области')
        elif self.mode == 'pan/zoom':
            self.set_message('Панорамирование осей')
            
    def release(self, event):
        """
        Обработка отпускания кнопки мыши, обеспечивает корректную работу
        завершения операций масштабирования и панорамирования.
        """
        # Вызываем родительский метод для обработки отпускания
        NavigationToolbar2Tk.release(self, event)
        
        # Обновляем сообщение после завершения операции
        if self.mode == 'zoom rect':
            self.set_message('Масштабирование области завершено')
        elif self.mode == 'pan/zoom':
            self.set_message('Панорамирование завершено')
        
        # Обновляем состояние кнопок
        self._update_buttons_checked()
        
    def mouse_move(self, event):
        """
        Обработка движения мыши, обеспечивает корректную работу
        с русскими подсказками во время операций.
        """
        # Вызываем родительский метод
        NavigationToolbar2Tk.mouse_move(self, event)
        
        # Переопределяем сообщения в зависимости от режима
        if self.mode == 'zoom rect' and self._lastCursor == 2:  # ZOOM
            self.set_message('Масштабирование области...')
        elif self.mode == 'pan/zoom' and self._lastCursor == 1:  # PAN
            self.set_message('Панорамирование...')
    
    def scroll_event(self, event):
        """
        Обработка прокрутки колёсика мыши для масштабирования.
        """
        # Вызываем родительский метод
        NavigationToolbar2Tk.scroll_event(self, event)
        
        # Обновляем сообщение
        self.set_message('Масштабирование колёсиком')

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