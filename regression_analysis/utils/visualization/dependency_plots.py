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
        # Обучаем модель множественной регрессии
        model = LinearRegression()
        model.fit(X, y)
        
        # Подготавливаем данные для графика
        y_pred = model.predict(X)
        
        # Строим график фактических и предсказанных значений
        ax.scatter(y, y_pred, color='blue', alpha=0.6, edgecolors='white')
        
        # Добавляем диагональную линию (идеальный прогноз)
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Настраиваем оси
        ax.set_xlabel(f'Фактический {target_name}', fontsize=11)
        ax.set_ylabel(f'Предсказанный {target_name}', fontsize=11)
        ax.set_title(f'Множественная регрессия: {target_name} и выбранные факторы', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем формулу множественной регрессии, упрощаем для длинных формул
        if n_features <= 3:
            equation = f"y = {model.intercept_:.2f}"
            for i, (coef, feature) in enumerate(zip(model.coef_, feature_names)):
                equation += f" + {coef:.2f}·{feature}"
        else:
            equation = f"y = {model.intercept_:.2f}"
            for i, (coef, feature) in enumerate(zip(model.coef_, feature_names)):
                if i < 2:  # Показываем только первые два коэффициента
                    equation += f" + {coef:.2f}·{feature}"
                elif i == 2:
                    equation += " + ..."
                    break
        
        # Расчет R²
        r2 = r2_score(y, y_pred)
        
        # Добавляем текст с уравнением и R² в верхний левый угол
        font_size = min(10, max(7, 12 - n_features * 0.5))  # Уменьшаем размер шрифта при большом количестве признаков
        ax.text(0.02, 0.98, equation, transform=ax.transAxes, fontsize=font_size, 
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        ax.text(0.02, 0.90, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=font_size,
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Добавляем аннотацию о переменных с возможностью прокрутки при большом количестве
        feature_text = "Переменные в модели:\n"
        max_features_to_show = min(n_features, 5)  # Показываем максимум 5 переменных
        for i, feature in enumerate(feature_names[:max_features_to_show]):
            feature_text += f"{i+1}. {feature}\n"
        if n_features > max_features_to_show:
            feature_text += f"...и еще {n_features - max_features_to_show} переменных"
        
        ax.text(0.02, 0.82, feature_text, transform=ax.transAxes, fontsize=max(7, font_size-1),
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Улучшаем макет и уменьшаем отступы
    fig.tight_layout(pad=0.5)
    return fig

def create_heatmap_plot(data, title="Корреляционная матрица", fig=None, ax=None):
    """
    Создает тепловую карту корреляций между переменными.
    
    Args:
        data (pd.DataFrame): DataFrame с данными для анализа корреляций
        title (str): Заголовок графика
        fig (matplotlib.figure.Figure): Существующая фигура (опционально)
        ax (matplotlib.axes.Axes): Существующая ось (опционально)
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с тепловой картой
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from ui.components.theme_manager import get_text_color_for_background, DARK_THEME
    
    # Рассчитываем корреляции
    corr_matrix = data.corr()
    
    # Создаем горизонтальную фигуру, если она не передана
    if fig is None:
        plt.figure(figsize=(12, 8))
        fig = plt.gcf()
    
    # Используем существующую ось или текущую ось фигуры
    if ax is None:
        ax = fig.add_subplot(111)
    
    # Создаем маску для верхнего треугольника
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
    
    # Получаем значения корреляции для использования в аннотациях
    annot_values = corr_matrix.values
    
    # Создаем матрицу цветов текста для каждой ячейки используя функцию из theme_manager
    text_colors = []
    for i in range(annot_values.shape[0]):
        row_colors = []
        for j in range(annot_values.shape[1]):
            if mask[i, j]:  # Если ячейка скрыта (верхний треугольник)
                row_colors.append('black')  # Значение не важно, так как оно будет скрыто
            else:
                # Используем улучшенную функцию определения цвета текста
                # Делаем цвета более контрастными
                if annot_values[i, j] > 0.5 or annot_values[i, j] < -0.5:
                    text_color = 'white'  # Для ярких цветов используем белый
                else:
                    text_color = 'black'  # Для светлых цветов используем черный
                row_colors.append(text_color)
        text_colors.append(row_colors)
    
    # Создаем свою цветовую карту с более яркими цветами для лучшего контраста
    colors = ["navy", "royalblue", "lightgray", "lightcoral", "darkred"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_coolwarm", colors, N=100)
    
    # Создаем тепловую карту с адаптивными цветами для текста
    sns.heatmap(
        corr_matrix,
        annot=True,              # Добавляем числовые значения
        fmt=".2f",               # Формат числовых значений (2 знака после запятой)
        cmap=custom_cmap,        # Используем кастомную цветовую карту для большего контраста
        mask=mask,               # Маска для верхнего треугольника
        linewidths=0.8,          # Увеличиваем ширину линий между ячейками для лучшей разделимости
        cbar_kws={"shrink": 0.6, "pad": 0.03},  # Настройки цветовой шкалы
        vmin=-1, vmax=1,         # Диапазон значений
        annot_kws={
            "size": 12,          # Увеличиваем размер текста для лучшей читаемости
            "weight": "bold"     # Делаем текст жирным для лучшей видимости
        },
        ax=ax                    # Используем переданную ось
    )
    
    # Изменяем цвета текста для каждой аннотации
    for i, j in zip(*np.where(~mask)):
        # Получаем индекс текстового элемента
        idx = i * len(corr_matrix) - int(i * (i + 1) / 2) + j - i - 1
        
        # Проверяем, что индекс в пределах допустимого диапазона
        if idx < len(ax.texts):
            text = ax.texts[idx]
            text.set_color(text_colors[i][j])
    
    # Настраиваем внешний вид
    ax.set_title(title, fontsize=16, pad=15, color=DARK_THEME['text_light'])
    
    # Увеличиваем отступы и улучшаем метки осей
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Оставляем место внизу для меток
    
    # Поворачиваем метки осей и увеличиваем их для лучшей читаемости
    # Используем 30 градусов для меток оси X для лучшей видимости
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Увеличиваем нижнее поле фигуры, чтобы поместились все метки
    plt.subplots_adjust(bottom=0.28, left=0.18)
    
    # Применяем стиль текста для меток осей для соответствия теме
    for label in ax.get_xticklabels():
        label.set_color(DARK_THEME['neutral'])
    for label in ax.get_yticklabels():
        label.set_color(DARK_THEME['neutral'])
    
    return fig 