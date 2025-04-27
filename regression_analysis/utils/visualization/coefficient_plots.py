"""
Модуль для визуализации коэффициентов регрессионных моделей.

Содержит функции для создания графиков и диаграмм, отображающих
коэффициенты регрессии, их значимость, важность и взаимосвязи.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from typing import Union, List, Tuple, Dict, Optional, Any, Callable

def create_coefficient_bar_plot(
    coefficients: np.ndarray,
    feature_names: List[str],
    p_values: Optional[np.ndarray] = None,
    title: str = "Коэффициенты регрессии",
    alpha: float = 0.05,
    horizontal: bool = True,
    max_features: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Создает столбчатую диаграмму коэффициентов регрессии с выделением значимых.
    
    Parameters:
    coefficients (np.ndarray): Массив коэффициентов регрессии
    feature_names (List[str]): Названия признаков
    p_values (np.ndarray, optional): p-значения для каждого коэффициента
    title (str): Заголовок графика
    alpha (float): Уровень значимости для выделения значимых коэффициентов
    horizontal (bool): Строить горизонтальную диаграмму
    max_features (int): Максимальное количество отображаемых признаков
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
    
    # Если есть константа (интерсепт), удаляем ее из визуализации
    if 'Константа' in feature_names or 'const' in feature_names or 'intercept' in feature_names:
        const_idx = feature_names.index('Константа' if 'Константа' in feature_names else 
                                      'const' if 'const' in feature_names else 'intercept')
        coefficients = np.delete(coefficients, const_idx)
        feature_names = feature_names.copy()  # Создаем копию, чтобы не изменять оригинал
        feature_names.pop(const_idx)
        if p_values is not None:
            p_values = np.delete(p_values, const_idx)
    
    # Сортируем коэффициенты по абсолютному значению
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    
    # Ограничиваем количество отображаемых признаков
    if len(sorted_indices) > max_features:
        sorted_indices = sorted_indices[:max_features]
    
    # Отсортированные данные
    sorted_coeffs = coefficients[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_p_values = p_values[sorted_indices] if p_values is not None else None
    
    # Определяем цвета в зависимости от знака и значимости
    colors = []
    for i, coef in enumerate(sorted_coeffs):
        if p_values is not None and sorted_p_values[i] < alpha:
            # Значимые коэффициенты
            color = '#28a745' if coef > 0 else '#dc3545'  # Зеленый для положительных, красный для отрицательных
        else:
            # Незначимые коэффициенты (более бледные)
            color = '#8eca98' if coef > 0 else '#e9a2ab'  # Бледно-зеленый для положительных, бледно-красный для отрицательных
        colors.append(color)
    
    # Строим диаграмму
    if horizontal:
        bars = ax.barh(sorted_names, sorted_coeffs, color=colors, alpha=0.8, edgecolor='white')
        
        # Добавляем вертикальную линию на нуле
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel('Значение коэффициента', fontsize=12, labelpad=10)
        ax.set_ylabel('Признак', fontsize=12, labelpad=10)
        
        # Добавляем значения и p-значения
        for i, bar in enumerate(bars):
            value = bar.get_width()
            text_color = 'black'
            
            # Добавляем значение коэффициента
            if abs(value) < 0.01:
                text = f"{value:.4f}"
            else:
                text = f"{value:.2f}"
            
            # Позиция текста зависит от знака коэффициента
            if value < 0:
                ax.text(value - max(abs(sorted_coeffs)) * 0.05, bar.get_y() + bar.get_height()/2, 
                       text, ha='right', va='center', color=text_color, fontweight='bold')
            else:
                ax.text(value + max(abs(sorted_coeffs)) * 0.05, bar.get_y() + bar.get_height()/2, 
                       text, ha='left', va='center', color=text_color, fontweight='bold')
            
            # Добавляем p-значение, если оно есть
            if p_values is not None:
                p_value = sorted_p_values[i]
                if p_value < 0.001:
                    p_text = "p < 0.001"
                elif p_value < 0.01:
                    p_text = f"p = {p_value:.3f}"
                else:
                    p_text = f"p = {p_value:.2f}"
                
                # Добавляем звездочки для обозначения значимости
                stars = ""
                if p_value < 0.001:
                    stars = "***"
                elif p_value < 0.01:
                    stars = "**"
                elif p_value < 0.05:
                    stars = "*"
                
                # Позиция p-значения
                if value < 0:
                    offset = max(abs(sorted_coeffs)) * 0.25
                    ax.text(value - offset, bar.get_y() + bar.get_height()/2, 
                           f"{p_text} {stars}", ha='right', va='center', 
                           color=text_color, fontsize=9, alpha=0.7)
                else:
                    offset = max(abs(sorted_coeffs)) * 0.25
                    ax.text(value + offset, bar.get_y() + bar.get_height()/2, 
                           f"{p_text} {stars}", ha='left', va='center', 
                           color=text_color, fontsize=9, alpha=0.7)
    else:
        bars = ax.bar(sorted_names, sorted_coeffs, color=colors, alpha=0.8, edgecolor='white')
        
        # Добавляем горизонтальную линию на нуле
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Подписи и заголовок
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.set_xlabel('Признак', fontsize=12, labelpad=10)
        ax.set_ylabel('Значение коэффициента', fontsize=12, labelpad=10)
        
        # Поворачиваем подписи на оси X
        ax.tick_params(axis='x', rotation=45)
        
        # Добавляем значения и p-значения
        for i, bar in enumerate(bars):
            value = bar.get_height()
            text_color = 'black'
            
            # Добавляем значение коэффициента
            if abs(value) < 0.01:
                text = f"{value:.4f}"
            else:
                text = f"{value:.2f}"
            
            # Позиция текста зависит от знака коэффициента
            if value < 0:
                ax.text(bar.get_x() + bar.get_width()/2, value - max(abs(sorted_coeffs)) * 0.05, 
                       text, ha='center', va='top', color=text_color, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, value + max(abs(sorted_coeffs)) * 0.05, 
                       text, ha='center', va='bottom', color=text_color, fontweight='bold')
    
    # Добавляем легенду для значимости
    if p_values is not None:
        labels = [
            plt.Rectangle((0, 0), 1, 1, color='#28a745', alpha=0.8, ec="white"),
            plt.Rectangle((0, 0), 1, 1, color='#dc3545', alpha=0.8, ec="white"),
            plt.Rectangle((0, 0), 1, 1, color='#8eca98', alpha=0.8, ec="white"),
            plt.Rectangle((0, 0), 1, 1, color='#e9a2ab', alpha=0.8, ec="white")
        ]
        
        legend_labels = [
            f"Значимые положительные (p < {alpha})",
            f"Значимые отрицательные (p < {alpha})",
            "Незначимые положительные",
            "Незначимые отрицательные"
        ]
        
        ax.legend(labels, legend_labels, loc='best', frameon=True, 
                 framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_importance_plot(
    coefficients: np.ndarray,
    feature_names: List[str],
    standardized_coefficients: Optional[np.ndarray] = None,
    title: str = "Относительная важность признаков",
    max_features: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Создает диаграмму относительной важности признаков в модели.
    
    Parameters:
    coefficients (np.ndarray): Массив коэффициентов регрессии
    feature_names (List[str]): Названия признаков
    standardized_coefficients (np.ndarray, optional): Стандартизированные коэффициенты (бета-коэффициенты)
    title (str): Заголовок графика
    max_features (int): Максимальное количество отображаемых признаков
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
    
    # Используем стандартизированные коэффициенты для оценки важности, если они предоставлены
    if standardized_coefficients is not None:
        importance = np.abs(standardized_coefficients)
    else:
        # Иначе используем абсолютные значения обычных коэффициентов
        importance = np.abs(coefficients)
    
    # Если есть константа, удаляем ее из визуализации
    if 'Константа' in feature_names or 'const' in feature_names or 'intercept' in feature_names:
        const_idx = feature_names.index('Константа' if 'Константа' in feature_names else 
                                      'const' if 'const' in feature_names else 'intercept')
        importance = np.delete(importance, const_idx)
        feature_names = feature_names.copy()  # Создаем копию, чтобы не изменять оригинал
        feature_names.pop(const_idx)
    
    # Сортируем по важности
    sorted_indices = np.argsort(importance)[::-1]
    
    # Ограничиваем количество отображаемых признаков
    if len(sorted_indices) > max_features:
        sorted_indices = sorted_indices[:max_features]
    
    # Отсортированные данные
    sorted_importance = importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Нормализуем важность для сравнения (относительная важность)
    if np.sum(sorted_importance) > 0:
        relative_importance = sorted_importance / np.sum(sorted_importance) * 100
    else:
        relative_importance = sorted_importance
    
    # Создаем цветовую схему для градиента
    cmap = cm.get_cmap('RdYlGn_r')  # Обратный красно-желто-зеленый градиент
    colors = [cmap(imp / max(relative_importance)) for imp in relative_importance]
    
    # Строим горизонтальную диаграмму
    bars = ax.barh(sorted_names, relative_importance, color=colors, alpha=0.8, edgecolor='white')
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Относительная важность (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Признак', fontsize=12, labelpad=10)
    
    # Добавляем значения
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f"{width:.1f}%", ha='left', va='center', fontweight='bold')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_significance_plot(
    coefficients: np.ndarray,
    p_values: np.ndarray,
    feature_names: List[str],
    title: str = "Значимость коэффициентов",
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Создает диаграмму рассеяния коэффициентов в зависимости от их p-значений.
    
    Parameters:
    coefficients (np.ndarray): Массив коэффициентов регрессии
    p_values (np.ndarray): Массив p-значений для коэффициентов
    feature_names (List[str]): Названия признаков
    title (str): Заголовок графика
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
    
    # Преобразуем p-значения в -log10(p), чтобы визуализировать малые p-значения
    log_p_values = -np.log10(p_values)
    
    # Масштабируем размер точек в зависимости от абсолютного значения коэффициентов
    sizes = 100 * np.abs(coefficients) / max(np.abs(coefficients)) + 50
    
    # Определяем цвета в зависимости от знака коэффициентов и значимости
    colors = []
    for i, coef in enumerate(coefficients):
        if p_values[i] < 0.05:
            # Значимые коэффициенты
            color = '#28a745' if coef > 0 else '#dc3545'  # Зеленый для положительных, красный для отрицательных
        else:
            # Незначимые коэффициенты (более бледные)
            color = '#8eca98' if coef > 0 else '#e9a2ab'  # Бледно-зеленый для положительных, бледно-красный для отрицательных
        colors.append(color)
    
    # Строим диаграмму рассеяния
    scatter = ax.scatter(np.abs(coefficients), log_p_values, c=colors, s=sizes, alpha=0.8, edgecolor='white')
    
    # Добавляем подписи к точкам
    for i, name in enumerate(feature_names):
        # Исключаем константу из подписей, если она есть
        if name in ['Константа', 'const', 'intercept']:
            continue
            
        # Не подписываем точки с очень маленькими коэффициентами
        if np.abs(coefficients[i]) < max(np.abs(coefficients)) * 0.1:
            continue
            
        ax.annotate(name, (np.abs(coefficients[i]), log_p_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Добавляем горизонтальные линии для порогов значимости
    significance_levels = [0.05, 0.01, 0.001]
    line_styles = ['--', '-.', ':']
    
    for i, level in enumerate(significance_levels):
        log_level = -np.log10(level)
        ax.axhline(y=log_level, color='gray', linestyle=line_styles[i], 
                  alpha=0.7, label=f'p = {level}')
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Абсолютное значение коэффициента', fontsize=12, labelpad=10)
    ax.set_ylabel('-log10(p-значение)', fontsize=12, labelpad=10)
    
    # Добавляем легенду для порогов значимости
    ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_correlation_matrix(
    correlation_matrix: np.ndarray,
    feature_names: List[str],
    title: str = "Корреляционная матрица",
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Создает тепловую карту корреляционной матрицы признаков.
    
    Parameters:
    correlation_matrix (np.ndarray): Корреляционная матрица
    feature_names (List[str]): Названия признаков
    title (str): Заголовок графика
    annotate (bool): Добавлять ли значения корреляции в ячейки
    figsize (tuple): Размер фигуры в дюймах
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    # Настраиваем общий стиль
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    
    # Создаем тепловую карту
    cmap = cm.get_cmap('RdBu_r')  # Красно-синий градиент (красный для отрицательных, синий для положительных)
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Отображаем значения корреляции в ячейках
    if annotate:
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                corr = correlation_matrix[i, j]
                
                # Определяем цвет текста в зависимости от значения корреляции
                text_color = 'white' if abs(corr) > 0.7 else 'black'
                
                # Не отображаем значения на диагонали (всегда 1.0)
                if i != j:
                    ax.text(j, i, f"{corr:.2f}", ha='center', va='center', color=text_color, fontsize=9)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    
    # Добавляем цветовую шкалу
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Коэффициент корреляции', fontsize=10)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_confidence_intervals_plot(
    coefficients: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    feature_names: List[str],
    p_values: Optional[np.ndarray] = None,
    title: str = "Доверительные интервалы коэффициентов",
    alpha: float = 0.05,
    max_features: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Создает график доверительных интервалов для коэффициентов регрессии.
    
    Parameters:
    coefficients (np.ndarray): Массив коэффициентов регрессии
    lower_ci (np.ndarray): Нижние границы доверительных интервалов
    upper_ci (np.ndarray): Верхние границы доверительных интервалов
    feature_names (List[str]): Названия признаков
    p_values (np.ndarray, optional): p-значения для коэффициентов
    title (str): Заголовок графика
    alpha (float): Уровень значимости для выделения значимых коэффициентов
    max_features (int): Максимальное количество отображаемых признаков
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
    
    # Сортируем коэффициенты по абсолютному значению
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    
    # Ограничиваем количество отображаемых признаков
    if len(sorted_indices) > max_features:
        sorted_indices = sorted_indices[:max_features]
    
    # Отсортированные данные
    sorted_coeffs = coefficients[sorted_indices]
    sorted_lower = lower_ci[sorted_indices]
    sorted_upper = upper_ci[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_p_values = p_values[sorted_indices] if p_values is not None else None
    
    # Определяем цвета и маркеры
    colors = []
    for i, coef in enumerate(sorted_coeffs):
        # Проверяем, пересекает ли доверительный интервал ноль
        crosses_zero = sorted_lower[i] <= 0 <= sorted_upper[i]
        
        if p_values is not None:
            is_significant = sorted_p_values[i] < alpha
            if is_significant:
                # Значимые коэффициенты
                color = '#28a745' if coef > 0 else '#dc3545'  # Зеленый для положительных, красный для отрицательных
            else:
                # Незначимые коэффициенты (более бледные)
                color = '#8eca98' if coef > 0 else '#e9a2ab'  # Бледно-зеленый для положительных, бледно-красный для отрицательных
        else:
            # Если p-значения не предоставлены, используем простую логику: пересекает ли доверительный интервал ноль
            if crosses_zero:
                color = '#8eca98' if coef > 0 else '#e9a2ab'  # Незначимые (бледные цвета)
            else:
                color = '#28a745' if coef > 0 else '#dc3545'  # Значимые (яркие цвета)
        
        colors.append(color)
    
    # Строим график "ошибок" (error bars)
    ax.errorbar(sorted_coeffs, np.arange(len(sorted_names)), 
               xerr=[(sorted_coeffs - sorted_lower), (sorted_upper - sorted_coeffs)], 
               fmt='o', ecolor='gray', capsize=5, capthick=1, markersize=8, 
               markerfacecolor=colors, markeredgecolor='white')
    
    # Добавляем вертикальную линию в нуле
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Настройка осей
    ax.set_yticks(np.arange(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Значение коэффициента', fontsize=12, labelpad=10)
    
    # Добавляем легенду для значимости
    if p_values is not None:
        labels = [
            plt.Rectangle((0, 0), 1, 1, fc='#28a745', ec="white"),
            plt.Rectangle((0, 0), 1, 1, fc='#dc3545', ec="white"),
            plt.Rectangle((0, 0), 1, 1, fc='#8eca98', ec="white"),
            plt.Rectangle((0, 0), 1, 1, fc='#e9a2ab', ec="white")
        ]
        
        legend_labels = [
            f"Значимые положительные (p < {alpha})",
            f"Значимые отрицательные (p < {alpha})",
            "Незначимые положительные",
            "Незначимые отрицательные"
        ]
        
        ax.legend(labels, legend_labels, loc='best', frameon=True, 
                 framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig