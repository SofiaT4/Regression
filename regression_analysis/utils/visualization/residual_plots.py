"""
Модуль для создания графиков анализа остатков регрессионных моделей.

Содержит функции для визуализации остатков модели, включая 
диаграммы рассеяния, гистограммы, QQ-графики и другие инструменты
для проверки допущений линейной регрессии.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import scipy.stats as stats
from typing import Union, List, Tuple, Dict, Optional, Any, Callable

def create_residual_scatter_plot(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str = "График остатков",
    x_label: str = "Предсказанные значения",
    y_label: str = "Остатки",
    standardize: bool = True,
    add_lines: bool = True,
    add_loess: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает график разброса остатков относительно предсказанных значений.
    
    Parameters:
    y_pred (np.ndarray): Предсказанные значения
    residuals (np.ndarray): Остатки (y_true - y_pred)
    title (str): Заголовок графика
    x_label (str): Подпись оси X
    y_label (str): Подпись оси Y
    standardize (bool): Стандартизировать ли остатки
    add_lines (bool): Добавлять ли горизонтальные линии на уровнях 0, +/-1.96
    add_loess (bool): Добавлять ли сглаженную кривую LOESS
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
    
    # Стандартизируем остатки, если требуется
    if standardize:
        residuals_std = residuals / np.std(residuals)
        y_label = "Стандартизированные остатки"
    else:
        residuals_std = residuals
    
    # Определяем цвета в зависимости от значения остатков
    colors = np.where(
        np.abs(residuals_std) > 1.96,
        '#dc3545',  # Красный для выбросов (|z| > 1.96)
        np.where(
            np.abs(residuals_std) > 1.0,
            '#fd7e14',  # Оранжевый для пограничных (1.0 < |z| < 1.96)
            '#28a745'   # Зеленый для нормальных (|z| < 1.0)
        )
    )
    
    # Размер точек зависит от величины остатка
    sizes = 30 + np.abs(residuals_std) * 10
    
    # Строим диаграмму рассеяния
    scatter = ax.scatter(y_pred, residuals_std, c=colors, s=sizes, 
                       alpha=0.8, edgecolor='white')
    
    # Добавляем горизонтальную линию на нуле
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Добавляем горизонтальные линии для уровней +/-1.96 и +/-1.0
    if add_lines and standardize:
        # Уровни +/-1.96 (95% доверительный интервал)
        ax.axhline(y=1.96, color='#dc3545', linestyle='--', alpha=0.5, 
                  label='95% CI (±1.96σ)')
        ax.axhline(y=-1.96, color='#dc3545', linestyle='--', alpha=0.5)
        
        # Уровни +/-1.0 (68% доверительный интервал)
        ax.axhline(y=1.0, color='#fd7e14', linestyle=':', alpha=0.5, 
                  label='68% CI (±1.0σ)')
        ax.axhline(y=-1.0, color='#fd7e14', linestyle=':', alpha=0.5)
    
    # Добавляем сглаженную кривую LOESS, если требуется
    if add_loess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Сортируем данные по оси X для корректного построения кривой
            sorted_idx = np.argsort(y_pred)
            sorted_x = y_pred[sorted_idx]
            sorted_y = residuals_std[sorted_idx]
            
            # Вычисляем сглаженную кривую
            z = lowess(sorted_y, sorted_x, frac=0.3, it=3, return_sorted=False)
            
            # Строим кривую
            ax.plot(sorted_x, z, color='#6f42c1', linestyle='-', linewidth=2, alpha=0.8, 
                   label='LOESS')
        except ImportError:
            print("Библиотека statsmodels не установлена. Кривая LOESS не будет добавлена.")
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    # Добавляем легенду
    if (add_lines and standardize) or add_loess:
        ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_residual_histogram(
    residuals: np.ndarray,
    title: str = "Распределение остатков",
    x_label: str = "Остатки",
    y_label: str = "Частота",
    standardize: bool = True,
    bins: int = 30,
    add_norm_curve: bool = True,
    add_rug: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Создает гистограмму распределения остатков.
    
    Parameters:
    residuals (np.ndarray): Остатки (y_true - y_pred)
    title (str): Заголовок графика
    x_label (str): Подпись оси X
    y_label (str): Подпись оси Y
    standardize (bool): Стандартизировать ли остатки
    bins (int): Количество корзин гистограммы
    add_norm_curve (bool): Добавлять ли кривую нормального распределения
    add_rug (bool): Добавлять ли "ковер" из точек внизу графика
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
    
    # Стандартизируем остатки, если требуется
    if standardize:
        residuals_std = residuals / np.std(residuals)
        x_label = "Стандартизированные остатки"
    else:
        residuals_std = residuals
    
    # Строим гистограмму
    # Вычисляем границы для корзин, чтобы захватить выбросы
    min_val = min(residuals_std.min(), -4) if standardize else residuals_std.min() * 1.1
    max_val = max(residuals_std.max(), 4) if standardize else residuals_std.max() * 1.1
    
    if standardize:
        # Для стандартизированных остатков используем равномерные корзины
        bin_edges = np.linspace(min_val, max_val, bins + 1)
    else:
        # Для нестандартизированных используем автоматическое определение
        bin_edges = bins
    
    # Цветовая схема для гистограммы
    if standardize:
        # Получаем границы корзин и центры для окрашивания
        n, bin_edges = np.histogram(residuals_std, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Определяем цвета для корзин в зависимости от расстояния от нуля
        colors = []
        for center in bin_centers:
            if abs(center) > 1.96:
                colors.append('#dc3545')  # Красный для выбросов
            elif abs(center) > 1.0:
                colors.append('#fd7e14')  # Оранжевый для пограничных значений
            else:
                colors.append('#28a745')  # Зеленый для нормальных значений
        
        # Строим гистограмму с окрашиванием корзин
        n, bins, patches = ax.hist(residuals_std, bins=bin_edges, alpha=0.8, edgecolor='white')
        
        # Окрашиваем корзины
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
    else:
        # Для нестандартизированных остатков просто используем синий цвет
        ax.hist(residuals_std, bins=bin_edges, color='#3366cc', alpha=0.8, edgecolor='white')
    
    # Добавляем кривую нормального распределения
    if add_norm_curve:
        # Создаем точки для кривой нормального распределения
        x = np.linspace(min_val, max_val, 100)
        
        # Параметры нормального распределения
        mean = np.mean(residuals_std)
        std = np.std(residuals_std)
        
        # Вычисляем функцию плотности вероятности
        pdf = stats.norm.pdf(x, mean, std)
        
        # Масштабируем высоту PDF к высоте гистограммы
        hist_height = np.max(np.histogram(residuals_std, bins=bin_edges)[0])
        pdf = pdf * hist_height / np.max(pdf)
        
        # Строим кривую
        ax.plot(x, pdf, color='#6f42c1', linestyle='-', linewidth=2, 
               label=f'Норм. распр. (μ={mean:.2f}, σ={std:.2f})')
    
    # Добавляем "ковер" из точек
    if add_rug:
        ax.plot(residuals_std, np.zeros_like(residuals_std) - 0.05 * hist_height, 
               '|', color='black', alpha=0.5, markersize=5)
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    # Добавляем легенду для кривой нормального распределения
    if add_norm_curve:
        ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Добавляем вертикальную линию на нуле
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Если остатки стандартизированы, добавляем линии на уровнях +/-1.96 и +/-1.0
    if standardize:
        # Уровни +/-1.96 (95% доверительный интервал)
        ax.axvline(x=1.96, color='#dc3545', linestyle='--', alpha=0.5)
        ax.axvline(x=-1.96, color='#dc3545', linestyle='--', alpha=0.5)
        
        # Уровни +/-1.0 (68% доверительный интервал)
        ax.axvline(x=1.0, color='#fd7e14', linestyle=':', alpha=0.5)
        ax.axvline(x=-1.0, color='#fd7e14', linestyle=':', alpha=0.5)
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_qq_plot(
    residuals: np.ndarray,
    title: str = "Q-Q график остатков",
    x_label: str = "Теоретические квантили",
    y_label: str = "Квантили выборки",
    standardize: bool = True,
    add_line: bool = True,
    add_confidence: bool = True,
    figsize: Tuple[int, int] = (8, 8)
) -> Figure:
    """
    Создает Q-Q график остатков для проверки нормальности распределения.
    
    Parameters:
    residuals (np.ndarray): Остатки (y_true - y_pred)
    title (str): Заголовок графика
    x_label (str): Подпись оси X
    y_label (str): Подпись оси Y
    standardize (bool): Стандартизировать ли остатки
    add_line (bool): Добавлять ли линию нормального распределения
    add_confidence (bool): Добавлять ли доверительную область
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
    
    # Стандартизируем остатки, если требуется
    if standardize:
        residuals_std = residuals / np.std(residuals)
    else:
        residuals_std = residuals
    
    # Сортируем остатки
    residuals_sorted = np.sort(residuals_std)
    
    # Теоретические квантили нормального распределения
    n = len(residuals_sorted)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    
    # Рисуем точки
    ax.scatter(theoretical_quantiles, residuals_sorted, color='#3366cc', 
              s=50, alpha=0.8, edgecolor='white')
    
    # Добавляем линию нормального распределения
    if add_line:
        # Определяем пределы для линии
        min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
        max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
        
        # Создаем линию с небольшим запасом
        line_x = np.array([min_val - 0.5, max_val + 0.5])
        line_y = line_x  # Линия y = x
        
        ax.plot(line_x, line_y, color='#dc3545', linestyle='-', linewidth=1.5, 
               label='Нормальное распределение')
    
    # Добавляем доверительную область (95%)
    if add_confidence:
        # Вычисляем 95% доверительные интервалы для каждого квантиля
        # на основе доверительных интервалов для порядковых статистик
        from scipy.stats import norm
        
        # Функция для расчета доверительных интервалов порядковых статистик
        def confidence_interval(p, n, confidence=0.95):
            # Вычисляем стандартную ошибку порядковой статистики
            se = np.sqrt(p * (1 - p) / n)
            
            # Вычисляем границы доверительного интервала
            alpha = 1 - confidence
            z = norm.ppf(1 - alpha / 2)
            
            lower = p - z * se
            upper = p + z * se
            
            # Ограничиваем интервал [0, 1]
            lower = max(0, lower)
            upper = min(1, upper)
            
            # Преобразуем обратно в квантили
            return norm.ppf(lower), norm.ppf(upper)
        
        # Вычисляем p для каждой порядковой статистики
        p = np.arange(1, n + 1) / (n + 1)
        
        # Вычисляем доверительные интервалы
        conf_intervals = [confidence_interval(pi, n) for pi in p]
        lower_ci = [ci[0] for ci in conf_intervals]
        upper_ci = [ci[1] for ci in conf_intervals]
        
        # Строим область доверительного интервала
        ax.fill_between(theoretical_quantiles, lower_ci, upper_ci, 
                       color='gray', alpha=0.2, label='95% доверительная область')
    
    # Подписи и заголовок
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    # Добавляем легенду
    if add_line or add_confidence:
        ax.legend(loc='best', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
    
    # Делаем оси одинакового масштаба
    ax.set_aspect('equal')
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Подгоняем макет
    fig.tight_layout()
    
    return fig

def create_standardized_residual_plot(
    X: Union[pd.DataFrame, np.ndarray],
    residuals: np.ndarray,
    feature_names: Optional[List[str]] = None,
    selected_features: Optional[List[int]] = None,
    title: str = "Зависимость остатков от признаков",
    standardize: bool = True,
    add_loess: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Figure:
    """
    Создает график зависимости остатков от значений предикторов (признаков).
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    residuals (np.ndarray): Остатки (y_true - y_pred)
    feature_names (List[str], optional): Названия признаков
    selected_features (List[int], optional): Индексы выбранных признаков (не более 4)
    title (str): Заголовок графика
    standardize (bool): Стандартизировать ли остатки
    add_loess (bool): Добавлять ли сглаженную кривую LOESS
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
    
    # Стандартизируем остатки, если требуется
    if standardize:
        residuals_std = residuals / np.std(residuals)
        y_label = "Стандартизированные остатки"
    else:
        residuals_std = residuals
        y_label = "Остатки"
    
    # Если названия признаков не предоставлены, создаем стандартные
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f'Признак {i+1}' for i in range(X.shape[1])]
    
    # Выбираем признаки для отображения
    if selected_features is None:
        # По умолчанию выбираем первые 4 признака или меньше
        selected_features = list(range(min(4, X.shape[1])))
    else:
        # Убеждаемся, что индексы корректны
        selected_features = [i for i in selected_features if i < X.shape[1]]
        # Ограничиваем количество до 4
        if len(selected_features) > 4:
            selected_features = selected_features[:4]
    
    # Определяем компоновку подграфиков
    n_features = len(selected_features)
    if n_features <= 2:
        nrows, ncols = 1, n_features
    else:
        nrows, ncols = 2, 2
    
    # Создаем фигуру
    fig = Figure(figsize=figsize, dpi=100, facecolor='#ffffff')
    
    # Создаем подграфики
    for i, feature_idx in enumerate(selected_features):
        if i < nrows * ncols:  # Защита от выхода за пределы
            ax = fig.add_subplot(nrows, ncols, i + 1)
            
            # Получаем данные для текущего признака
            if hasattr(X, 'iloc'):
                x_data = X.iloc[:, feature_idx]
            else:
                x_data = X[:, feature_idx]
            
            # Определяем цвета в зависимости от значения остатков
            colors = np.where(
                np.abs(residuals_std) > 1.96,
                '#dc3545',  # Красный для выбросов (|z| > 1.96)
                np.where(
                    np.abs(residuals_std) > 1.0,
                    '#fd7e14',  # Оранжевый для пограничных (1.0 < |z| < 1.96)
                    '#28a745'   # Зеленый для нормальных (|z| < 1.0)
                )
            )
            
            # Размер точек зависит от величины остатка
            sizes = 30 + np.abs(residuals_std) * 5
            
            # Строим диаграмму рассеяния
            ax.scatter(x_data, residuals_std, c=colors, s=sizes, 
                      alpha=0.8, edgecolor='white')
            
            # Добавляем горизонтальную линию на нуле
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Добавляем горизонтальные линии для уровней +/-1.96 и +/-1.0
            if standardize:
                # Уровни +/-1.96 (95% доверительный интервал)
                ax.axhline(y=1.96, color='#dc3545', linestyle='--', alpha=0.5)
                ax.axhline(y=-1.96, color='#dc3545', linestyle='--', alpha=0.5)
                
                # Уровни +/-1.0 (68% доверительный интервал)
                ax.axhline(y=1.0, color='#fd7e14', linestyle=':', alpha=0.5)
                ax.axhline(y=-1.0, color='#fd7e14', linestyle=':', alpha=0.5)
            
            # Добавляем сглаженную кривую LOESS, если требуется
            if add_loess:
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    
                    # Сортируем данные по оси X для корректного построения кривой
                    sorted_idx = np.argsort(x_data)
                    sorted_x = x_data[sorted_idx]
                    sorted_y = residuals_std[sorted_idx]
                    
                    # Вычисляем сглаженную кривую
                    z = lowess(sorted_y, sorted_x, frac=0.6, it=3, return_sorted=False)
                    
                    # Строим кривую
                    ax.plot(sorted_x, z, color='#6f42c1', linestyle='-', linewidth=2, alpha=0.8)
                except ImportError:
                    print("Библиотека statsmodels не установлена. Кривая LOESS не будет добавлена.")
            
            # Подписи и заголовок
            ax.set_title(feature_names[feature_idx], fontsize=12)
            ax.set_xlabel("Значение признака", fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            
            # Удаляем лишние рамки
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Общий заголовок для всей фигуры
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Подгоняем макет
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Оставляем место для общего заголовка
    
    return fig