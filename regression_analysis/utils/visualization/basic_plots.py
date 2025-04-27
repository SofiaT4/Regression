"""
Модуль с функциями для создания базовых графиков и диаграмм.

Содержит функции для визуализации данных с помощью основных типов графиков,
таких как линейные графики, диаграммы рассеяния, столбчатые диаграммы и т.д.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from typing import Union, List, Tuple, Dict, Optional, Any, Callable

def create_actual_vs_predicted_plot(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: np.ndarray, 
    title: str = "Фактический и прогнозируемый ВВП",
    years: Optional[Union[pd.Series, np.ndarray, List[int]]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает график сравнения фактических и предсказанных значений.
    
    Parameters:
    y_true (pd.Series или np.ndarray): Фактические значения
    y_pred (np.ndarray): Предсказанные значения
    title (str): Заголовок графика
    years (pd.Series, np.ndarray или List[int], optional): Годы для оси X
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Создаем основные настройки визуального стиля
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Если годы не предоставлены, используем последовательные целые числа
    x_values = years if years is not None else range(len(y_true))
    x_label = 'Год' if years is not None else 'Порядковый номер наблюдения'
    
    # Строим график
    ax.plot(x_values, y_true, marker='o', color='#3366cc', linewidth=2, 
            markersize=6, label='Фактические значения')
    ax.plot(x_values, y_pred, linestyle='--', color='#dc3545', linewidth=2, 
            label='Прогнозные значения')
    
    # Добавляем подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel('Значение', fontsize=12, labelpad=10)
    
    # Настраиваем легенду и форматирование осей
    ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    ax.tick_params(axis='x', rotation=45)
    
    # Форматирование чисел с пробелами между тысячами
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', ' ')))
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_scatter_plot(
    x: Union[pd.Series, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    title: str = "Диаграмма рассеяния",
    x_label: str = "X", 
    y_label: str = "Y",
    color: str = '#3366cc',
    marker_size: int = 60,
    add_trendline: bool = True,
    annotations: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает диаграмму рассеяния с дополнительными опциями.
    
    Parameters:
    x (pd.Series или np.ndarray): Значения для оси X
    y (pd.Series или np.ndarray): Значения для оси Y
    title (str): Заголовок графика
    x_label (str): Подпись оси X
    y_label (str): Подпись оси Y
    color (str): Цвет точек
    marker_size (int): Размер маркеров
    add_trendline (bool): Добавлять ли линию тренда
    annotations (List[str], optional): Подписи к точкам
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Настраиваем общий стиль
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Строим диаграмму рассеяния
    scatter = ax.scatter(x, y, c=color, s=marker_size, alpha=0.8, edgecolor='white')
    
    # Добавляем линию тренда, если нужно
    if add_trendline and len(x) > 1:
        try:
            # Полиномиальная аппроксимация первого порядка (линейная)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Вычисляем значения для линии тренда
            x_line = np.linspace(min(x), max(x), 100)
            y_line = p(x_line)
            
            # Добавляем линию тренда
            ax.plot(x_line, y_line, color='#dc3545', linestyle='--', 
                   linewidth=2, label=f'y = {z[0]:.4f}x + {z[1]:.4f}')
            
            # Добавляем легенду
            ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
        except Exception as e:
            print(f"Не удалось добавить линию тренда: {e}")
    
    # Добавляем подписи к точкам, если предоставлены
    if annotations is not None and len(annotations) == len(x):
        for i, txt in enumerate(annotations):
            ax.annotate(txt, (x[i], y[i]), xytext=(5, 5), 
                      textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Добавляем подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_line_plot(
    x: Union[pd.Series, np.ndarray], 
    y_values: Dict[str, Union[pd.Series, np.ndarray]], 
    title: str = "Линейный график",
    x_label: str = "X", 
    y_label: str = "Y",
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает линейный график для нескольких рядов данных.
    
    Parameters:
    x (pd.Series или np.ndarray): Значения для оси X
    y_values (Dict[str, pd.Series или np.ndarray]): Словарь {название_ряда: значения}
    title (str): Заголовок графика
    x_label (str): Подпись оси X
    y_label (str): Подпись оси Y
    colors (Dict[str, str], optional): Словарь {название_ряда: цвет}
    markers (Dict[str, str], optional): Словарь {название_ряда: маркер}
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Определяем цвета по умолчанию
    default_colors = {
        'series1': '#3366cc', 
        'series2': '#dc3545', 
        'series3': '#28a745', 
        'series4': '#fd7e14', 
        'series5': '#6f42c1'
    }
    # Определяем маркеры по умолчанию
    default_markers = {
        'series1': 'o', 
        'series2': 's', 
        'series3': '^', 
        'series4': 'D', 
        'series5': 'x'
    }
    
    # Если не указаны пользовательские цвета или маркеры, используем значения по умолчанию
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    
    # Настраиваем общий стиль
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Строим графики для каждого ряда данных
    for i, (name, y) in enumerate(y_values.items()):
        series_key = f'series{i+1}'
        color = colors.get(name, default_colors.get(series_key, '#3366cc'))
        marker = markers.get(name, default_markers.get(series_key, 'o'))
        
        ax.plot(x, y, marker=marker, color=color, linewidth=2, 
               markersize=6, label=name)
    
    # Добавляем подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    # Настраиваем легенду
    ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Форматируем оси
    ax.tick_params(axis='x', rotation=45)
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_bar_plot(
    categories: Union[pd.Series, np.ndarray, List[str]], 
    values: Union[pd.Series, np.ndarray, List[float]], 
    title: str = "Столбчатая диаграмма",
    x_label: str = "Категории", 
    y_label: str = "Значения",
    color: str = '#3366cc',
    horizontal: bool = False,
    show_values: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает столбчатую диаграмму.
    
    Parameters:
    categories (pd.Series, np.ndarray, или List[str]): Категории для оси
    values (pd.Series, np.ndarray, или List[float]): Значения для отображения
    title (str): Заголовок графика
    x_label (str): Подпись оси X (или Y для горизонтальной диаграммы)
    y_label (str): Подпись оси Y (или X для горизонтальной диаграммы)
    color (str): Цвет столбцов
    horizontal (bool): Строить горизонтальную диаграмму
    show_values (bool): Отображать значения над столбцами
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Настраиваем общий стиль
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Определяем цвета с чередованием оттенков для лучшей визуализации
    if len(values) > 1:
        colors = [color] * len(values)
        for i in range(1, len(values), 2):
            # Слегка осветляем каждый второй столбец
            h = color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            lighter_color = '#{:02x}{:02x}{:02x}'.format(
                min(255, int(rgb[0] * 1.2)),
                min(255, int(rgb[1] * 1.2)),
                min(255, int(rgb[2] * 1.2))
            )
            colors[i] = lighter_color
    else:
        colors = [color]
    
    # Строим диаграмму
    if horizontal:
        bars = ax.barh(categories, values, color=colors, alpha=0.8, edgecolor='white')
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel(y_label, fontsize=12, labelpad=10)
        ax.set_ylabel(x_label, fontsize=12, labelpad=10)
    else:
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white')
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        
        # Поворачиваем подписи на оси X, если их много
        if len(categories) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы, если требуется
    if show_values:
        for bar in bars:
            height = bar.get_height() if not horizontal else bar.get_width()
            
            # Форматируем текст зависимости от величины числа
            if abs(height) >= 1000000:
                text = f"{height/1000000:.1f}M"
            elif abs(height) >= 1000:
                text = f"{height/1000:.1f}k"
            elif abs(height) >= 1:
                text = f"{height:.1f}"
            else:
                text = f"{height:.3f}"
            
            # Располагаем текст в зависимости от ориентации
            if horizontal:
                ax.text(height + (max(values) * 0.02), bar.get_y() + bar.get_height()/2, 
                       text, ha='left', va='center', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, height + (max(values) * 0.02), 
                       text, ha='center', va='bottom', fontweight='bold')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_grouped_bar_plot(
    categories: Union[pd.Series, np.ndarray, List[str]], 
    group_data: Dict[str, Union[pd.Series, np.ndarray, List[float]]], 
    title: str = "Сгруппированная столбчатая диаграмма",
    x_label: str = "Категории", 
    y_label: str = "Значения",
    colors: Optional[Dict[str, str]] = None,
    horizontal: bool = False,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Создает сгруппированную столбчатую диаграмму для нескольких групп данных.
    
    Parameters:
    categories (pd.Series, np.ndarray, или List[str]): Категории для оси
    group_data (Dict[str, pd.Series/np.ndarray/List[float]]): Словарь {название_группы: значения}
    title (str): Заголовок графика
    x_label (str): Подпись оси X (или Y для горизонтальной диаграммы)
    y_label (str): Подпись оси Y (или X для горизонтальной диаграммы)
    colors (Dict[str, str], optional): Словарь {название_группы: цвет}
    horizontal (bool): Строить горизонтальную диаграмму
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Определяем цвета по умолчанию
    default_colors = {
        'group1': '#3366cc', 
        'group2': '#dc3545', 
        'group3': '#28a745', 
        'group4': '#fd7e14', 
        'group5': '#6f42c1'
    }
    
    # Если не указаны пользовательские цвета, используем значения по умолчанию
    if colors is None:
        colors = {}
    
    # Настраиваем общий стиль
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Определяем ширину столбца и расположение групп
    n_groups = len(categories)
    n_bars = len(group_data)
    bar_width = 0.8 / n_bars
    
    # Создаем позиции для групп столбцов
    if horizontal:
        # Для горизонтальной диаграммы
        positions = np.arange(n_groups)
        
        for i, (group_name, values) in enumerate(group_data.items()):
            # Вычисляем смещение для текущей группы
            pos = positions + (i - n_bars/2 + 0.5) * bar_width
            
            # Выбираем цвет для группы
            group_key = f'group{i+1}'
            color = colors.get(group_name, default_colors.get(group_key, '#3366cc'))
            
            # Строим горизонтальные столбцы
            ax.barh(pos, values, height=bar_width, color=color, 
                   alpha=0.8, edgecolor='white', label=group_name)
        
        # Устанавливаем метки категорий на оси Y
        ax.set_yticks(positions)
        ax.set_yticklabels(categories)
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel(y_label, fontsize=12, labelpad=10)
        ax.set_ylabel(x_label, fontsize=12, labelpad=10)
    
    else:
        # Для вертикальной диаграммы
        positions = np.arange(n_groups)
        
        for i, (group_name, values) in enumerate(group_data.items()):
            # Вычисляем смещение для текущей группы
            pos = positions + (i - n_bars/2 + 0.5) * bar_width
            
            # Выбираем цвет для группы
            group_key = f'group{i+1}'
            color = colors.get(group_name, default_colors.get(group_key, '#3366cc'))
            
            # Строим вертикальные столбцы
            ax.bar(pos, values, width=bar_width, color=color, 
                  alpha=0.8, edgecolor='white', label=group_name)
        
        # Устанавливаем метки категорий на оси X
        ax.set_xticks(positions)
        ax.set_xticklabels(categories)
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        
        # Поворачиваем подписи на оси X, если их много
        if len(categories) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Добавляем легенду
    ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig