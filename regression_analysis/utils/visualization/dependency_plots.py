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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.inspection import partial_dependence
from matplotlib.gridspec import GridSpec

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
    
    # Создаем фигуру
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = fig.gca()
    
    # Создаем диаграмму рассеяния
    if year_data is not None:
        # Если есть данные о годах, используем их для цветовой кодировки
        scatter = ax.scatter(
            x_data, y_data, 
            c=year_data, 
            cmap='viridis', 
            s=80, 
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
                    s=120, 
                    edgecolors='red', 
                    facecolors='none', 
                    linewidth=2, 
                    label=f'Год {highlight_year}'
                )
        
        # Добавляем цветовую шкалу для годов
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('Год', fontsize=12)
    else:
        ax.scatter(x_data, y_data, color='blue', s=80, alpha=0.7, edgecolors='w')
    
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
        ax.plot(x_range, y_pred, 'r-', linewidth=2)
        
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
            
            ax.text(0.02, 0.95, equation, transform=ax.transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    else:
        # Линейная регрессия
        coef = np.polyfit(x_data, y_data, 1)
        poly1d_fn = np.poly1d(coef)
        ax.plot(x_data, poly1d_fn(x_data), 'r-', linewidth=2)
        
        # Показываем уравнение регрессии
        if show_equation:
            equation = f"y = {coef[1]:.2f} + {coef[0]:.2f}x"
            ax.text(0.02, 0.95, equation, transform=ax.transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Добавляем R²
    y_pred = poly1d_fn(x_data) if not fit_polynomial else model.predict(X)
    r2 = r2_score(y_data, y_pred)
    ax.text(0.02, 0.89, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=12,
          verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Настраиваем оси и заголовок
    ax.set_xlabel(feature_name, fontsize=14)
    ax.set_ylabel("ВВП", fontsize=14)
    ax.set_title(f"Зависимость ВВП от {feature_name}", fontsize=16)
    
    # Добавляем сетку
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Если есть выделенный год, добавляем легенду
    if year_data is not None and highlight_year is not None and highlight_year in year_data:
        ax.legend()
    
    # Улучшаем внешний вид
    fig.tight_layout()
    
    return fig

def create_multi_scatter_plot(X, y, feature_names, target_name="GDP"):
    """
    Создает множественную диаграмму рассеяния для нескольких признаков.
    
    Args:
        X (pd.DataFrame): DataFrame с данными признаков
        y (pd.Series): Целевая переменная
        feature_names (list): Список названий признаков
        target_name (str): Название целевой переменной
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с множественной диаграммой рассеяния
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Определяем размер сетки графиков
    n_features = len(feature_names)
    
    if n_features == 1:
        # Если только один признак, создаем обычную диаграмму рассеяния
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X[feature_names[0]], y, color='blue', alpha=0.6, edgecolors='white')
        
        # Добавляем линию тренда
        coef = np.polyfit(X[feature_names[0]], y, 1)
        poly1d_fn = np.poly1d(coef)
        x_range = np.linspace(X[feature_names[0]].min(), X[feature_names[0]].max(), 100)
        ax.plot(x_range, poly1d_fn(x_range), 'r--', linewidth=2)
        
        # Настраиваем оси
        ax.set_xlabel(feature_names[0], fontsize=14)
        ax.set_ylabel(target_name, fontsize=14)
        ax.set_title(f'Зависимость {target_name} от {feature_names[0]}', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем формулу тренда
        equation = f"y = {coef[0]:.2f}x + {coef[1]:.2f}"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    else:
        # Создаем сетку графиков
        rows = (n_features + 1) // 2  # Округление вверх для нечетного числа признаков
        cols = min(2, n_features)  # Максимум 2 колонки
        
        if n_features > 4:
            # Если признаков больше 4, размещаем в 2 колонки
            rows = (n_features + 1) // 2
            cols = 2
        elif n_features == 4:
            # Для 4 признаков делаем сетку 2x2
            rows = 2
            cols = 2
        elif n_features == 3:
            # Для 3 признаков делаем сетку 2x2, но используем только 3 ячейки
            rows = 2
            cols = 2
        else:  # n_features == 2
            # Для 2 признаков делаем сетку 1x2
            rows = 1
            cols = 2
        
        # Создаем фигуру и сетку
        fig = plt.figure(figsize=(cols * 6, rows * 4))
        
        if n_features == 3:
            # Для 3 признаков используем GridSpec для более гибкой компоновки
            gs = GridSpec(2, 2, figure=fig)
            axes = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, :])
            ]
        else:
            # Для остальных случаев обычная сетка
            axes = [fig.add_subplot(rows, cols, i+1) for i in range(n_features)]
        
        # Создаем графики для каждого признака
        for i, feature in enumerate(feature_names):
            if i < len(axes):  # Защита от выхода за границы списка осей
                ax = axes[i]
                
                # Диаграмма рассеяния
                ax.scatter(X[feature], y, color='blue', alpha=0.6, edgecolors='white')
                
                # Добавляем линию тренда
                coef = np.polyfit(X[feature], y, 1)
                poly1d_fn = np.poly1d(coef)
                x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
                ax.plot(x_range, poly1d_fn(x_range), 'r--', linewidth=2)
                
                # Настраиваем оси
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel(target_name, fontsize=12)
                ax.set_title(f'{target_name} vs {feature}', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Добавляем формулу тренда
                equation = f"y = {coef[0]:.2f}x + {coef[1]:.2f}"
                ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, 
                       verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Добавляем общий заголовок
    fig.suptitle(f'Зависимость {target_name} от различных факторов', fontsize=16, y=1.02)
    
    # Улучшаем макет
    fig.tight_layout()
    return fig

def create_3d_plot(X, y, feature_names, target_name="GDP", years=None):
    """
    Создает трехмерный график зависимости целевой переменной от двух признаков.
    
    Args:
        X (pd.DataFrame): DataFrame с двумя колонками - признаками для осей X и Z
        y (np.ndarray | pd.Series): Целевая переменная для оси Y
        feature_names (list): Названия признаков [x_name, z_name]
        target_name (str): Название целевой переменной
        years (list): Список годов для цветовой кодировки точек
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с 3D графиком
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Извлекаем названия признаков
    x_feature = feature_names[0]
    z_feature = feature_names[1]
    
    # Извлекаем данные
    x1 = X[x_feature].values
    x2 = X[z_feature].values
    
    # Создаем сетку для поверхности
    x1_grid = np.linspace(min(x1), max(x1), 20)
    x2_grid = np.linspace(min(x2), max(x2), 20)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    
    # Обучаем регрессионную модель
    model = LinearRegression()
    model.fit(X, y)
    
    # Предсказываем значения для сетки
    grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])
    grid_df = pd.DataFrame(grid_points, columns=[x_feature, z_feature])
    y_pred_grid = model.predict(grid_df)
    zz = y_pred_grid.reshape(xx1.shape)
    
    # Строим поверхность регрессии
    surf = ax.plot_surface(xx1, xx2, zz, alpha=0.5, cmap='viridis', 
                         rstride=1, cstride=1, linewidth=0, antialiased=True)
    
    # Добавляем точки данных
    if years is not None and len(years) == len(y):
        # Используем годы для цветовой кодировки
        scatter = ax.scatter(x1, x2, y, c=years, cmap='cool', s=50, alpha=0.8, edgecolors='w')
        fig.colorbar(scatter, ax=ax, pad=0.1, label='Год')
    else:
        ax.scatter(x1, x2, y, c='r', s=50, alpha=0.8, edgecolors='w')
    
    # Добавляем название осей
    ax.set_xlabel(x_feature, fontsize=14)
    ax.set_ylabel(z_feature, fontsize=14)
    ax.set_zlabel(target_name, fontsize=14)
    
    # Добавляем заголовок и формулу регрессии
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"{target_name} = {intercept:.2f} + {coefficients[0]:.2f}×{x_feature} + {coefficients[1]:.2f}×{z_feature}"
    ax.set_title(f"Зависимость {target_name} от {x_feature} и {z_feature}\n{equation}", fontsize=16)
    
    # Добавляем информацию о R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    fig.text(0.02, 0.02, f"R² = {r2:.3f}", fontsize=12)
    
    # Улучшаем внешний вид
    fig.tight_layout()
    
    return fig

def create_heatmap_plot(data, title="Корреляционная матрица"):
    """
    Создает тепловую карту корреляций между переменными.
    
    Args:
        data (pd.DataFrame): DataFrame с данными для анализа корреляций
        title (str): Заголовок графика
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с тепловой картой
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Рассчитываем корреляции
    corr_matrix = data.corr()
    
    # Создаем фигуру
    plt.figure(figsize=(10, 8))
    fig = plt.gcf()
    
    # Создаем тепловую карту с seaborn
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
    
    # Создаем тепловую карту
    ax = sns.heatmap(
        corr_matrix, 
        annot=True,             # Добавляем числовые значения
        fmt=".2f",              # Формат числовых значений (2 знака после запятой)
        cmap="coolwarm",        # Цветовая карта
        mask=mask,              # Маска для верхнего треугольника
        linewidths=0.5,         # Ширина линий между ячейками
        cbar_kws={"shrink": 0.8}, # Настройки цветовой шкалы
        vmin=-1, vmax=1         # Диапазон значений
    )
    
    # Настраиваем внешний вид
    plt.title(title, fontsize=16, pad=20)
    
    # Поворачиваем метки осей для лучшей читаемости
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Добавляем пояснение к графику
    fig.text(0.02, 0.02, (
        "Значения показывают коэффициенты корреляции Пирсона между переменными.\n"
        "Диапазон от -1 (сильная отрицательная корреляция) до 1 (сильная положительная корреляция)."
    ), fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_partial_dependence_plot(model, X, feature_idx, feature_name, target_name="GDP"):
    """
    Создает график частичной зависимости целевой переменной от конкретного признака.
    
    Args:
        model: Обученная модель sklearn
        X: Матрица признаков, используемая для обучения модели
        feature_idx: Индекс признака, для которого строится график частичной зависимости
        feature_name: Название признака
        target_name: Название целевой переменной
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры с графиком частичной зависимости
    """
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Рассчитываем частичную зависимость
    feature_values = X.iloc[:, feature_idx].unique() if hasattr(X, 'iloc') else np.unique(X[:, feature_idx])
    feature_values.sort()
    
    # Для непрерывных признаков создаем более плотную сетку
    if len(feature_values) > 10:
        grid_resolution = 50
        feature_values = np.linspace(
            np.min(feature_values),
            np.max(feature_values),
            num=grid_resolution
        )
    
    # Рассчитываем частичную зависимость
    try:
        # Для scikit-learn >= 0.22
        pdp_result = partial_dependence(
            model, X, [feature_idx], 
            grid_resolution=len(feature_values) if len(feature_values) < 50 else 50,
            kind='average'
        )
        pdp_values = pdp_result['average'][0]
        pdp_feature_values = pdp_result['values'][0]
    except TypeError:
        # Для более старых версий scikit-learn
        pdp_feature_values, pdp_values = partial_dependence(
            model, X, [feature_idx], 
            grid_resolution=len(feature_values) if len(feature_values) < 50 else 50
        )
        pdp_values = pdp_values[0]
        pdp_feature_values = pdp_feature_values[0]
    
    # Строим график
    ax.plot(pdp_feature_values, pdp_values, '-', color='#1f77b4', linewidth=2.5)
    ax.scatter(pdp_feature_values, pdp_values, color='#1f77b4', s=30)
    
    # Добавляем полиномиальную аппроксимацию для более наглядного отображения тренда
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    
    # Создаем полиномиальную регрессию 3 степени
    X_poly = pdp_feature_values.reshape(-1, 1)
    y_poly = pdp_values
    
    # Строим модель полиномиальной регрессии
    poly_degree = min(3, len(pdp_feature_values) - 1)  # Не более 3, но не больше чем кол-во точек - 1
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(X_poly, y_poly)
    
    # Создаем более плотную сетку для отрисовки кривой
    X_plot = np.linspace(min(pdp_feature_values), max(pdp_feature_values), 100).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    # Рисуем линию тренда
    ax.plot(X_plot, y_plot, '--', color='#ff7f0e', linewidth=2, label=f'Тренд (полином {poly_degree} степени)')
    
    # Находим точки максимума и минимума на тренде
    max_idx = np.argmax(y_plot)
    min_idx = np.argmin(y_plot)
    
    # Отмечаем точки максимума и минимума
    ax.scatter(X_plot[max_idx], y_plot[max_idx], color='green', s=80, marker='^', 
              label=f'Максимум: {X_plot[max_idx][0]:.2f}')
    ax.scatter(X_plot[min_idx], y_plot[min_idx], color='red', s=80, marker='v', 
              label=f'Минимум: {X_plot[min_idx][0]:.2f}')
    
    # Добавляем среднее значение целевой переменной как горизонтальную линию
    ax.axhline(y=np.mean(pdp_values), color='gray', linestyle='--', alpha=0.7,
              label=f'Среднее {target_name}')
    
    # Оформление графика
    ax.set_xlabel(feature_name, fontsize=14)
    ax.set_ylabel(f'{target_name}', fontsize=14)
    ax.set_title(f'Частичная зависимость {target_name} от {feature_name}', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=12)
    
    # Добавляем пояснение к графику
    fig.text(0.02, 0.02, (
        f"График показывает, как изменяется {target_name} при изменении {feature_name},\n"
        f"когда все остальные признаки остаются постоянными."
    ), fontsize=10, alpha=0.7)
    
    fig.tight_layout()
    return fig 