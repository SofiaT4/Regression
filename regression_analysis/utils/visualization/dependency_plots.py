"""
Модуль с функциями для создания графиков зависимости ВВП от других экономических показателей.

Содержит функции для визуализации зависимостей ВВП от различных факторов,
таких как безработица и численность занятых в разных возрастных группах.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, List, Tuple, Dict, Optional, Any
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.colors as mcolors
from ui.components.theme_manager import get_text_color_for_background, DARK_THEME
import matplotlib.patheffects as PathEffects

def create_scatter_plot(
    x_data,
    y_data,
    feature_name,
    year_data=None,
    highlight_year=None,
    fig=None,
    fit_polynomial=False,
    polynomial_degree=2,
    show_equation=True
):
    """
    Создает диаграмму рассеяния для зависимости целевой переменной от одного признака.
    
    Args:
        x_data (pd.Series | np.ndarray): Данные для оси X
        y_data (pd.Series | np.ndarray): Целевая переменная (ВВП и т.п.)
        feature_name (str): Название признака
        year_data (list): Список годов для цветовой кодировки точек
        highlight_year (int): Год для выделения
        fig (matplotlib.figure.Figure): Существующая фигура (опционально)
        fit_polynomial (bool): Добавить полиномиальную регрессию
        polynomial_degree (int): Степень полинома
        show_equation (bool): Показывать уравнение регрессии
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с диаграммой рассеяния
    """
    import matplotlib.patches as mpatches
    
    # Создаем фигуру с горизонтальным соотношением сторон
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.gca()
    
    # Создаем диаграмму рассеяния
    if year_data is not None:
        # Если есть данные о годах, используем их для цветовой кодировки
        scatter = ax.scatter(
            x_data, y_data, 
            c=year_data, 
            cmap='viridis', 
            s=60, 
            alpha=0.7, 
            edgecolors='w'
        )
        
        # Если нужно выделить конкретный год
        if highlight_year is not None and highlight_year in year_data:
            # Находим индексы для выделяемого года
            highlight_indices = [i for i, year in enumerate(year_data) if year == highlight_year]
            
            # Выделяем точки для указанного года
            if highlight_indices:
                x_highlight = [x_data.iloc[i] if hasattr(x_data, 'iloc') else x_data[i] for i in highlight_indices]
                y_highlight = [y_data.iloc[i] if hasattr(y_data, 'iloc') else y_data[i] for i in highlight_indices]
                
                ax.scatter(
                    x_highlight, y_highlight, 
                    s=100, 
                    edgecolors='red', 
                    facecolors='none', 
                    linewidth=2, 
                    label=f'Год {highlight_year}'
                )
        
        # Добавляем цветовую шкалу для годов
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, shrink=0.7)
        cbar.set_label('Год', fontsize=10)
    else:
        ax.scatter(x_data, y_data, color='blue', s=60, alpha=0.7, edgecolors='w')
    
    # Добавляем линию тренда
    if fit_polynomial:
        # Полиномиальная регрессия
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        
        # Подготавливаем данные
        X = x_data.values.reshape(-1, 1) if hasattr(x_data, 'values') else x_data.reshape(-1, 1)
        
        # Создаем и обучаем модель
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=polynomial_degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X, y_data)
        
        # Создаем точки для линии тренда
        x_range = np.linspace(min(x_data), max(x_data), 100)
        X_range = x_range.reshape(-1, 1)
        y_pred = model.predict(X_range)
        
        # Строим линию тренда
        ax.plot(x_range, y_pred, 'r-', linewidth=1.5)
        
        # Показываем уравнение регрессии
        if show_equation:
            coefs = model.named_steps['linear'].coef_
            intercept = model.named_steps['linear'].intercept_
            
            equation = f"y = {intercept:.2f}"
            for i, coef in enumerate(coefs[1:]):
                if i == 0:
                    equation += f" + {coef:.2f}x"
                else:
                    equation += f" + {coef:.2f}x^{i+1}"
            
            ax.text(0.02, 0.95, equation, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    else:
        # Линейная регрессия
        coef = np.polyfit(x_data, y_data, 1)
        poly1d_fn = np.poly1d(coef)
        ax.plot(x_data, poly1d_fn(x_data), 'r-', linewidth=1.5)
        
        # Показываем уравнение регрессии
        if show_equation:
            equation = f"y = {coef[1]:.2f} + {coef[0]:.2f}x"
            ax.text(0.02, 0.95, equation, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Добавляем R²
    y_pred = poly1d_fn(x_data) if not fit_polynomial else model.predict(X)
    r2 = r2_score(y_data, y_pred)
    ax.text(0.02, 0.89, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=10,
          verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Настраиваем оси и заголовок
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel("ВВП", fontsize=11)
    ax.set_title(f"Зависимость ВВП от {feature_name}", fontsize=12)
    
    # Добавляем сетку
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Если есть выделенный год, добавляем легенду
    if year_data is not None and highlight_year is not None and highlight_year in year_data:
        ax.legend(fontsize=9)
    
    # Улучшаем внешний вид
    fig.tight_layout(pad=0.5)
    
    return fig

def create_multi_scatter_plot(X, y, feature_names, target_name="GDP", fig=None, ax=None):
    """
    Создает множественную диаграмму рассеяния для нескольких признаков.
    
    Args:
        X (pd.DataFrame): DataFrame с данными признаков
        y (pd.Series): Целевая переменная
        feature_names (list): Список названий признаков
        target_name (str): Название целевой переменной
        fig (matplotlib.figure.Figure): Существующая фигура (опционально)
        ax (matplotlib.axes.Axes): Существующая ось (опционально)
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с множественной диаграммой рассеяния
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.linear_model import LinearRegression
    
    # Определяем размер сетки графиков
    n_features = len(feature_names)
    
    # Создаем фигуру, если она не была передана
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    if n_features == 1:
        # Если только один признак, создаем обычную диаграмму рассеяния с горизонтальным соотношением сторон
        ax.scatter(X[feature_names[0]], y, color='blue', alpha=0.6, edgecolors='white')
        
        # Добавляем линию тренда
        coef = np.polyfit(X[feature_names[0]], y, 1)
        poly1d_fn = np.poly1d(coef)
        x_range = np.linspace(X[feature_names[0]].min(), X[feature_names[0]].max(), 100)
        ax.plot(x_range, poly1d_fn(x_range), 'r--', linewidth=2)
        
        # Настраиваем оси
        ax.set_xlabel(feature_names[0], fontsize=12)
        ax.set_ylabel(target_name, fontsize=12)
        ax.set_title(f'Зависимость {target_name} от {feature_names[0]}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем формулу тренда
        equation = f"y = {coef[0]:.2f}x + {coef[1]:.2f}"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    else:
        # Если выбрано несколько переменных, ось X — сумма значений выбранных переменных, подпись — их список
        X_sum = X[feature_names].sum(axis=1)
        features_str = ", ".join(feature_names)
        ax.scatter(X_sum, y, color='blue', alpha=0.6, edgecolors='white')
        
        # Добавляем линию тренда
        coef = np.polyfit(X_sum, y, 1)
        poly1d_fn = np.poly1d(coef)
        x_range = np.linspace(X_sum.min(), X_sum.max(), 100)
        ax.plot(x_range, poly1d_fn(x_range), 'r--', linewidth=2)
        
        # Настраиваем оси
        ax.set_xlabel(features_str, fontsize=11)
        ax.set_ylabel(target_name, fontsize=11)
        ax.set_title(f'Зависимость {target_name} от суммы: {features_str}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем формулу тренда
        equation = f"y = {coef[0]:.2f}x + {coef[1]:.2f}"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Улучшаем макет и уменьшаем отступы
    fig.tight_layout(pad=0.5)
    return fig

def create_heatmap_plot(data, title="Корреляционная матрица", fig=None, ax=None):
    """
    Строит тепловую карту корреляций между экономическими показателями с шаблонными подписями.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ui.components.theme_manager import DARK_THEME
    import matplotlib.patheffects as PathEffects
    import pandas as pd

    # Сопоставление шаблонных названий с возможными вариантами в данных
    column_map = {
        "ВВП": ["ввп", "gdp", "валовой", "ввп в текущих ценах"],
        "Безработные": ["безработные", "численность безработных", "unemploy"],
        "25-34": ["25-34", "25_34", "25–34"],
        "35-44": ["35-44", "35_44", "35–44"],
        "45-49": ["45-49", "45_49", "45–49"],
        "50-59": ["50-59", "50_59", "50–59"],
    }

    # Приводим названия столбцов к нижнему регистру для поиска
    data_columns = {col.lower(): col for col in data.columns}

    # Формируем новый DataFrame с шаблонными названиями
    selected = {}
    for template, variants in column_map.items():
        for variant in variants:
            for col_lower, col_orig in data_columns.items():
                if variant in col_lower:
                    selected[template] = data[col_orig]
                    break
            if template in selected:
                break

    # Оставляем только найденные столбцы, но в нужном порядке
    ordered_keys = [k for k in column_map.keys() if k in selected]
    if len(ordered_keys) < 2:
        raise ValueError("Недостаточно данных для построения корреляционной карты: нужно минимум 2 показателя из шаблона.")

    corr_data = pd.DataFrame({k: selected[k] for k in ordered_keys})

    # Считаем корреляцию
    corr_matrix = corr_data.corr()

    # Строим тепловую карту
    if fig is None:
        plt.figure(figsize=(8, 6))
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "pad": 0.03},
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    for text in ax.texts:
        text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    ax.set_title("Корреляция между экономическими показателями", fontsize=16, pad=15, color=DARK_THEME['text_light'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(bottom=0.40, left=0.18)

    for label in ax.get_xticklabels():
        label.set_color(DARK_THEME['neutral'])
    for label in ax.get_yticklabels():
        label.set_color(DARK_THEME['neutral'])

    return fig 