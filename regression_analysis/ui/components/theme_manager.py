"""
Модуль управления темами для приложения.
Содержит цветовые схемы и функции для применения стилей к Tkinter-интерфейсу.
"""

import tkinter as tk
from tkinter import ttk

# Определение цветовой схемы темной темы
DARK_THEME = {
    'primary': "#1A1A1A",     # угольный для фона и меню
    'accent': "#6C8EBF",      # приглушенный синий для интерактива
    'secondary': "#800000",   # классический бордовый для акцентов и графиков
    'neutral': "#CCCCCC",     # светло-серый для текста
    'bg': "#262626",          # темно-серый для контраста
    'bg_light': "#333333",    # немного светлее для элементов
    'highlight': "#404040",   # для подсветки выбранных элементов
    'error': "#661a1a",       # для ошибок 
    'success': "#1a662a",     # для успешных операций
    'text_light': "#EEEEEE",  # светлый текст
    'text_dark': "#999999",   # приглушенный текст
    'border': "#444444"       # цвет границ
}

# Активная тема
CURRENT_THEME = DARK_THEME

def apply_theme(root):
    """
    Применяет текущую тему ко всему приложению.
    
    Parameters:
    root (tk.Tk or tk.Toplevel): Корневое окно приложения
    """
    # Применяем тему к ttk виджетам
    _apply_ttk_styles(root)
    # Применяем тему к стандартным виджетам
    _apply_tk_options(root)

def _apply_ttk_styles(root):
    """Настраивает стили ttk виджетов"""
    style = ttk.Style(root)
    style.theme_use('clam')  # Базовая тема, хорошо работает с темными цветами
    
    # Treeview (таблицы)
    style.configure("Treeview", 
                  background=CURRENT_THEME['bg'], 
                  foreground=CURRENT_THEME['neutral'],
                  fieldbackground=CURRENT_THEME['primary'],
                  borderwidth=0)
    style.map('Treeview', 
             background=[('selected', CURRENT_THEME['accent'])],
             foreground=[('selected', CURRENT_THEME['text_light'])])
    
    # Заголовки Treeview
    style.configure("Treeview.Heading", 
                  background=CURRENT_THEME['bg_light'], 
                  foreground=CURRENT_THEME['text_light'],
                  relief="flat")
    style.map("Treeview.Heading",
              background=[('active', CURRENT_THEME['accent'])],
              foreground=[('active', CURRENT_THEME['text_light'])])
    
    # Кнопки
    style.configure("TButton", 
                  background=CURRENT_THEME['bg_light'],
                  foreground=CURRENT_THEME['neutral'],
                  borderwidth=1,
                  focusthickness=3,
                  focuscolor=CURRENT_THEME['accent'])
    style.map("TButton",
              background=[('active', CURRENT_THEME['accent']), 
                          ('pressed', CURRENT_THEME['secondary'])],
              foreground=[('active', CURRENT_THEME['text_light']), 
                          ('pressed', CURRENT_THEME['text_light'])])
    
    # Акцентные кнопки
    style.configure("Accent.TButton", 
                  background=CURRENT_THEME['accent'],
                  foreground=CURRENT_THEME['text_light'])
    style.map("Accent.TButton",
              background=[('active', CURRENT_THEME['secondary']), 
                          ('pressed', CURRENT_THEME['secondary'])],
              foreground=[('active', CURRENT_THEME['text_light']), 
                          ('pressed', CURRENT_THEME['text_light'])])
    
    # Вторичные кнопки
    style.configure("Secondary.TButton", 
                  background=CURRENT_THEME['secondary'],
                  foreground=CURRENT_THEME['text_light'])
    style.map("Secondary.TButton",
              background=[('active', CURRENT_THEME['accent']), 
                          ('pressed', CURRENT_THEME['accent'])],
              foreground=[('active', CURRENT_THEME['text_light']), 
                          ('pressed', CURRENT_THEME['text_light'])])
    
    # Метки
    style.configure("TLabel", 
                   background=CURRENT_THEME['primary'],
                   foreground=CURRENT_THEME['neutral'])
    
    # Поля ввода
    style.configure("TEntry", 
                   fieldbackground=CURRENT_THEME['bg'],
                   foreground=CURRENT_THEME['text_light'],
                   bordercolor=CURRENT_THEME['border'],
                   lightcolor=CURRENT_THEME['accent'],
                   darkcolor=CURRENT_THEME['accent'])
    
    # Combobox (выпадающие списки)
    style.configure("TCombobox",
                  background=CURRENT_THEME['bg'],
                  foreground=CURRENT_THEME['text_light'],
                  fieldbackground=CURRENT_THEME['bg'],
                  selectbackground=CURRENT_THEME['accent'],
                  selectforeground=CURRENT_THEME['text_light'],
                  arrowcolor=CURRENT_THEME['text_light'],
                  bordercolor=CURRENT_THEME['border'])
    
    # Применяем стили к отдельным компонентам Combobox
    style.map('TCombobox', 
             fieldbackground=[('readonly', CURRENT_THEME['bg'])],
             background=[('readonly', CURRENT_THEME['bg']),
                        ('active', CURRENT_THEME['bg_light'])],
             foreground=[('readonly', CURRENT_THEME['text_light'])],
             selectbackground=[('readonly', CURRENT_THEME['accent'])],
             selectforeground=[('readonly', CURRENT_THEME['text_light'])])
    
    # Настраиваем выпадающий список Combobox
    root.option_add('*TCombobox*Listbox.background', CURRENT_THEME['bg'])
    root.option_add('*TCombobox*Listbox.foreground', CURRENT_THEME['text_light'])
    root.option_add('*TCombobox*Listbox.selectBackground', CURRENT_THEME['accent'])
    root.option_add('*TCombobox*Listbox.selectForeground', CURRENT_THEME['text_light'])
    
    # Рамки
    style.configure("TFrame", background=CURRENT_THEME['primary'])
    style.configure("TLabelframe", 
                   background=CURRENT_THEME['primary'],
                   foreground=CURRENT_THEME['neutral'],
                   bordercolor=CURRENT_THEME['border'],
                   darkcolor=CURRENT_THEME['accent'],
                   lightcolor=CURRENT_THEME['accent'])
    style.configure("TLabelframe.Label", 
                   background=CURRENT_THEME['primary'],
                   foreground=CURRENT_THEME['neutral'])
    
    # Notebook (вкладки)
    style.configure("TNotebook", 
                   background=CURRENT_THEME['primary'],
                   tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", 
                   background=CURRENT_THEME['bg_light'],
                   foreground=CURRENT_THEME['neutral'],
                   padding=[10, 2],
                   borderwidth=0)
    style.map("TNotebook.Tab",
              background=[('selected', CURRENT_THEME['accent']), 
                          ('active', CURRENT_THEME['bg'])],
              foreground=[('selected', CURRENT_THEME['text_light']), 
                          ('active', CURRENT_THEME['text_light'])])
    
    # Полосы прокрутки
    style.configure("Vertical.TScrollbar", 
                   background=CURRENT_THEME['bg_light'],
                   troughcolor=CURRENT_THEME['primary'],
                   arrowcolor=CURRENT_THEME['neutral'],
                   bordercolor=CURRENT_THEME['border'])
    style.map("Vertical.TScrollbar",
              background=[('active', CURRENT_THEME['accent']), 
                          ('pressed', CURRENT_THEME['secondary'])])
    
    style.configure("Horizontal.TScrollbar", 
                   background=CURRENT_THEME['bg_light'],
                   troughcolor=CURRENT_THEME['primary'],
                   arrowcolor=CURRENT_THEME['neutral'],
                   bordercolor=CURRENT_THEME['border'])
    style.map("Horizontal.TScrollbar",
              background=[('active', CURRENT_THEME['accent']), 
                          ('pressed', CURRENT_THEME['secondary'])])
    
    # Checkbutton и Radiobutton
    style.configure("TCheckbutton", 
                   background=CURRENT_THEME['primary'],
                   foreground=CURRENT_THEME['neutral'])
    style.map("TCheckbutton",
             background=[('active', CURRENT_THEME['primary'])],
             foreground=[('active', CURRENT_THEME['text_light'])])
    
    style.configure("TRadiobutton", 
                   background=CURRENT_THEME['primary'],
                   foreground=CURRENT_THEME['neutral'])
    style.map("TRadiobutton",
             background=[('active', CURRENT_THEME['primary'])],
             foreground=[('active', CURRENT_THEME['text_light'])])

def _apply_tk_options(root):
    """Настраивает опции для стандартных виджетов Tkinter"""
    # Настройка цвета фона основного окна
    root.configure(bg=CURRENT_THEME['primary'])
    
    # Настройка стандартных виджетов Tkinter
    root.option_add("*Background", CURRENT_THEME['primary'])
    root.option_add("*Foreground", CURRENT_THEME['neutral'])
    root.option_add("*Font", ("Arial", 10))
    
    # Кнопки
    root.option_add("*Button.Background", CURRENT_THEME['bg_light'])
    root.option_add("*Button.Foreground", CURRENT_THEME['neutral'])
    root.option_add("*Button.activeBackground", CURRENT_THEME['accent'])
    root.option_add("*Button.activeForeground", CURRENT_THEME['text_light'])
    
    # Метки
    root.option_add("*Label.Background", CURRENT_THEME['primary'])
    root.option_add("*Label.Foreground", CURRENT_THEME['neutral'])
    
    # Поля ввода
    root.option_add("*Entry.Background", CURRENT_THEME['bg'])
    root.option_add("*Entry.Foreground", CURRENT_THEME['text_light'])
    root.option_add("*Entry.selectBackground", CURRENT_THEME['accent'])
    root.option_add("*Entry.selectForeground", CURRENT_THEME['text_light'])
    
    # Многострочные поля ввода
    root.option_add("*Text.Background", CURRENT_THEME['bg'])
    root.option_add("*Text.Foreground", CURRENT_THEME['text_light'])
    
    # Выпадающие списки
    root.option_add("*Listbox.Background", CURRENT_THEME['bg'])
    root.option_add("*Listbox.Foreground", CURRENT_THEME['text_light'])
    root.option_add("*Listbox.selectBackground", CURRENT_THEME['accent'])
    
    # Холсты
    root.option_add("*Canvas.Background", CURRENT_THEME['primary'])

def style_treeview_tags(treeview):
    """
    Применяет стили к тегам Treeview для выделения важных строк.
    
    Parameters:
    treeview (ttk.Treeview): Treeview для стилизации
    """
    treeview.tag_configure("header", background=CURRENT_THEME['bg_light'], 
                         foreground=CURRENT_THEME['text_light'], font=('Arial', 9, 'bold'))
    treeview.tag_configure("subheader", background=CURRENT_THEME['bg'])
    treeview.tag_configure("best", background="#225522")  # Темно-зеленый для лучших показателей
    treeview.tag_configure("worst", background="#552222")  # Темно-красный для худших показателей
    treeview.tag_configure("separator", background=CURRENT_THEME['bg_light'])
    treeview.tag_configure("significant", background="#225522")  # Темно-зеленый для значимых
    treeview.tag_configure("not_significant", background="#552222")  # Темно-красный для незначимых

def get_chart_colors():
    """
    Возвращает словарь с цветами для графиков matplotlib.
    
    Returns:
    dict: Словарь с цветами для различных элементов графиков
    """
    return {
        'figure.facecolor': CURRENT_THEME['primary'],
        'axes.facecolor': CURRENT_THEME['bg'],
        'axes.edgecolor': CURRENT_THEME['neutral'],
        'axes.labelcolor': CURRENT_THEME['neutral'],
        'axes.titlecolor': CURRENT_THEME['neutral'],
        'text.color': CURRENT_THEME['neutral'],
        'xtick.color': CURRENT_THEME['neutral'],
        'ytick.color': CURRENT_THEME['neutral'],
        'grid.color': CURRENT_THEME['bg_light'],
        'grid.alpha': 0.5,
        'lines.color': CURRENT_THEME['accent'],
        'patch.facecolor': CURRENT_THEME['accent'],
        'boxplot.boxprops.color': CURRENT_THEME['neutral'],
        'boxplot.medianprops.color': CURRENT_THEME['accent'],
        'scatter.edgecolors': CURRENT_THEME['neutral']
    }

def get_text_color_for_background(val, colormap_name='coolwarm'):
    """
    Определяет оптимальный цвет текста для фона на основе значения и цветовой карты.
    
    Для светлого фона возвращает темный текст, для темного фона - светлый текст.
    
    Args:
        val (float): Значение для нормализации в диапазоне цветовой карты.
            Для карты 'coolwarm' предполагается диапазон от -1 до 1.
        colormap_name (str): Название цветовой карты matplotlib.
    
    Returns:
        str: Цвет текста ('black' или 'white')
    """
    import matplotlib.pyplot as plt
    
    # Получаем объект цветовой карты
    cmap = plt.cm.get_cmap(colormap_name)
    
    # Нормализуем значение к диапазону 0-1 в зависимости от карты
    if colormap_name == 'coolwarm':
        # Для coolwarm ожидаем значения от -1 до 1
        norm_val = (val + 1) / 2
    else:
        # Для других карт просто ограничиваем диапазон от 0 до 1
        norm_val = max(0, min(1, val))
    
    # Получаем RGB цвет фона
    bg_color = cmap(norm_val)[:3]  # Первые три значения - это RGB
    
    # Рассчитываем яркость цвета по формуле (0.299*R + 0.587*G + 0.114*B)
    brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    
    # Возвращаем черный или белый в зависимости от яркости
    # Если яркость > 0.5, фон светлый, используем черный текст
    # Если яркость <= 0.5, фон темный, используем белый текст
    return 'black' if brightness > 0.5 else 'white'

def apply_chart_style(plt):
    """
    Применяет темную тему к графикам matplotlib.
    
    Parameters:
    plt: Модуль matplotlib.pyplot
    """
    chart_colors = get_chart_colors()
    for key, value in chart_colors.items():
        plt.rcParams[key] = value