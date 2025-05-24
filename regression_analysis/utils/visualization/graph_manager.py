"""
Модуль для создания и управления графиками регрессионного анализа.

Содержит функции для создания различных типов графиков на основе
результатов регрессионного анализа, включая графики фактических и
предсказанных значений, визуализацию коэффициентов, графики остатков
и динамики показателей.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Настройка логирования
logger = logging.getLogger(__name__)

def create_graph(
    graph_index: int, 
    df: pd.DataFrame, 
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    model: LinearRegression, 
    y_pred: np.ndarray, 
    model_type: str,
    **kwargs
) -> Figure:
    """
    Создает график на основе индекса типа графика и типа модели.
    
    Parameters:
    graph_index (int): Индекс типа графика
    df (pandas.DataFrame): Датафрейм с данными
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (LinearRegression): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    model_type (str): Тип модели ('all_groups', 'unemployed', 'combined')
    **kwargs: Дополнительные параметры для настройки графика
    
    Returns:
    Figure: Объект фигуры matplotlib
    """
    logger.info(f"Создание графика типа {graph_index} для модели {model_type}")
    
    # Импортируем настройки темы
    from ui.components.theme_manager import DARK_THEME, apply_chart_style
    
    # Применяем темный стиль для matplotlib
    apply_chart_style(plt)
    
    # Создаем фигуру с цветом фона из темы
    fig = Figure(figsize=(10, 6), dpi=100, facecolor=DARK_THEME['primary'])
    
    # Форматируем уравнение регрессии для отображения
    coefficients = [model.intercept_]
    coefficients.extend(model.coef_)
    
    # Названия моделей для заголовков
    model_names = {
        'all_groups': 'численности рабочих',
        'unemployed': 'безработицы',
        'combined': 'численности рабочих и безработицы'
    }
    
    # Проверяем, что колонка с годом существует
    if 'Год' not in df.columns:
        # Создаем последовательность лет, начиная с 2000 года
        years = range(2000, 2000 + len(df))
        x_label = 'Порядковый номер наблюдения'
    else:
        years = df['Год']
        x_label = 'Год'
    
    try:
        if graph_index == 0:
            # Фактический и прогнозируемый ВВП
            fig = create_actual_predicted_plot(fig, years, y, y_pred, model_type, model_names, 
                                            coefficients, x_label, X)  # Передаем X
            
        elif graph_index == 1:
            # Визуализация коэффициентов модели
            max_features = kwargs.get('max_features', None)
            horizontal = kwargs.get('horizontal', True)
            fig = create_coefficients_plot(fig, X, model, model_type, horizontal, max_features)
            
        elif graph_index == 2:
            # График остатков
            fig = create_residuals_plot(fig, years, y, y_pred, model_type, model_names)
            
        else:  # graph_index == 3
            # Динамика показателей
            selected_features = kwargs.get('selected_features', None)
            fig = create_dynamics_plot(fig, years, y, X, model_type, model_names, selected_features)
    
    except Exception as e:
        logger.error(f"Ошибка при создании графика {graph_index}: {str(e)}")
        # В случае ошибки создаем пустой график с сообщением об ошибке
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Ошибка при создании графика:\n{str(e)}", 
               ha='center', va='center', fontsize=12, color=DARK_THEME['error'])
        ax.axis('off')
        
        # Применяем стиль к осям
        ax.set_facecolor(DARK_THEME['bg'])
    
    # Для каждой оси на графике применяем темный стиль
    for ax in fig.get_axes():
        ax.set_facecolor(DARK_THEME['bg'])
        ax.xaxis.label.set_color(DARK_THEME['neutral'])
        ax.yaxis.label.set_color(DARK_THEME['neutral'])
        ax.title.set_color(DARK_THEME['neutral'])
        ax.tick_params(colors=DARK_THEME['neutral'])
        
        # Настраиваем цвет линий осей
        for spine in ax.spines.values():
            spine.set_color(DARK_THEME['neutral'])
    
    fig.tight_layout()
    return fig

def create_actual_predicted_plot(
    fig: Figure, 
    years: Union[pd.Series, np.ndarray, range], 
    y: Union[pd.Series, np.ndarray], 
    y_pred: np.ndarray, 
    model_type: str, 
    model_names: Dict[str, str], 
    coefficients: List[float], 
    x_label: str,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> Figure:
    """
    Создает график фактических и прогнозируемых значений ВВП.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    years: Значения для оси X (годы или номера наблюдений)
    y: Фактические значения ВВП
    y_pred: Предсказанные значения ВВП
    model_type: Тип модели ('all_groups', 'unemployed', 'combined')
    model_names: Словарь с названиями моделей для заголовков
    coefficients: Коэффициенты модели регрессии
    x_label: Подпись оси X
    X: Признаки модели (опциональный параметр)
    
    Returns:
    Figure: Обновленный объект фигуры matplotlib
    """
    # Импортируем настройки темы
    from ui.components.theme_manager import DARK_THEME
    
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_THEME['bg'])
    
    # Преобразуем Series в numpy массивы для однородности
    if isinstance(y, pd.Series):
        y = y.values
    
    # Рассчитываем R² корректно для отображения
    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_valid) > 0:
        y_mean = np.mean(y_valid)
        ss_total = np.sum((y_valid - y_mean) ** 2)
        ss_residual = np.sum((y_valid - y_pred_valid) ** 2)
        
        if ss_total > 0:
            r2 = 1 - (ss_residual / ss_total)
        else:
            r2 = 0
    else:
        r2 = 0
    
    # Цвета из темы для графиков
    actual_color = DARK_THEME['accent']
    predicted_color = DARK_THEME['secondary']
    
    # Строим графики
    ax.plot(years, y, marker='o', color=actual_color, linewidth=2, markersize=6, label='Фактический ВВП')
    ax.plot(years, y_pred, linestyle='--', color=predicted_color, linewidth=2, label='Прогнозируемый ВВП')

    # Явно задаем xticks и xticklabels для целых годов
    if hasattr(years, 'min') and hasattr(years, 'max'):
        year_list = list(range(int(years.min()), int(years.max()) + 1))
    else:
        year_list = list(sorted(set(int(y) for y in years)))
    ax.set_xticks(year_list)
    ax.set_xticklabels([str(y) for y in year_list])
    
    # Заголовок с R²
    ax.set_title(f'Фактический и прогнозируемый ВВП (модель от {model_names[model_type]})\nR² = {r2:.4f}', 
                 fontsize=14, pad=20, fontweight='bold', color=DARK_THEME['neutral'])
    ax.set_xlabel(x_label, fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
    ax.set_ylabel('ВВП', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
    
    # Формирование уравнения в зависимости от типа модели
    if model_type == 'unemployed':
        if len(coefficients) >= 2:
            equation = f"y = {format_coef(coefficients[1])}·Безработица + {format_coef(coefficients[0])}"
        else:
            equation = f"y = {format_coef(coefficients[0])}"
    elif model_type == 'all_groups':
        if len(coefficients) > 1:
            # Формируем уравнение для возрастных групп
            equation = f"y = {format_coef(coefficients[0])}"
            
            # Добавляем до 3 наиболее важных коэффициентов
            if len(coefficients) > 4:  # Если много коэффициентов
                # Сортируем по абсолютной величине, исключая константу
                sorted_indices = sorted(range(1, len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)
                
                # Выбираем до 3 наиболее важных коэффициентов
                for i, idx in enumerate(sorted_indices[:3]):
                    coef = coefficients[idx]
                    if coef >= 0:
                        equation += f" + {format_coef(coef)}·X{idx}"
                    else:
                        equation += f" - {format_coef(abs(coef))}·X{idx}"
                
                if len(sorted_indices) > 3:
                    equation += " + ..."
            else:
                # Если коэффициентов мало, показываем все
                for i in range(1, len(coefficients)):
                    coef = coefficients[i]
                    if coef >= 0:
                        equation += f" + {format_coef(coef)}·X{i}"
                    else:
                        equation += f" - {format_coef(abs(coef))}·X{i}"
        else:
            equation = f"y = {format_coef(coefficients[0])}"
    else:  # combined
        if len(coefficients) > 2:  # Если есть и коэффициенты возрастных групп, и безработицы
            # Считаем, что последний коэффициент - это безработица
            unemployment_idx = len(coefficients) - 1
            unemployment_coef = coefficients[unemployment_idx]
            
            # Находим наиболее значимую возрастную группу
            if unemployment_idx > 1:
                age_indices = list(range(1, unemployment_idx))
                age_idx = max(age_indices, key=lambda i: abs(coefficients[i]))
                age_coef = coefficients[age_idx]
                
                # Формируем уравнение с возрастной группой и безработицей
                equation = f"y = {format_coef(coefficients[0])}"
                
                # Добавляем коэффициент возрастной группы
                if age_coef >= 0:
                    equation += f" + {format_coef(age_coef)}·Возр.гр."
                else:
                    equation += f" - {format_coef(abs(age_coef))}·Возр.гр."
                
                # Добавляем коэффициент безработицы
                if unemployment_coef >= 0:
                    equation += f" + {format_coef(unemployment_coef)}·Безраб."
                else:
                    equation += f" - {format_coef(abs(unemployment_coef))}·Безраб."
                
                # Добавляем многоточие, если есть еще много коэффициентов
                if len(coefficients) > 4:
                    equation += " + ..."
            else:
                # Если нет возрастных групп, только безработица
                equation = f"y = {format_coef(coefficients[0])}"
                
                if unemployment_coef >= 0:
                    equation += f" + {format_coef(unemployment_coef)}·Безработица"
                else:
                    equation += f" - {format_coef(abs(unemployment_coef))}·Безработица"
        elif len(coefficients) == 2:  # Только один коэффициент кроме константы
            equation = f"y = {format_coef(coefficients[0])}"
            
            if coefficients[1] >= 0:
                equation += f" + {format_coef(coefficients[1])}·X"
            else:
                equation += f" - {format_coef(abs(coefficients[1]))}·X"
        else:
            equation = f"y = {format_coef(coefficients[0])}"
    
    # Добавляем уравнение на график с темным фоном
    ax.text(0.02, 0.95, equation, transform=ax.transAxes, fontsize=10, color=DARK_THEME['neutral'],
            bbox=dict(facecolor=DARK_THEME['bg_light'], alpha=0.8, boxstyle='round,pad=0.5', edgecolor=DARK_THEME['border']))
    
    # Настраиваем легенду с цветами из темы
    legend = ax.legend(loc='best')
    legend.get_frame().set_facecolor(DARK_THEME['bg_light'])
    legend.get_frame().set_edgecolor(DARK_THEME['border'])
    
    for text in legend.get_texts():
        text.set_color(DARK_THEME['neutral'])
    
    ax.tick_params(axis='x', rotation=45, colors=DARK_THEME['neutral'])
    ax.tick_params(axis='y', colors=DARK_THEME['neutral'])
    
    # Форматирование чисел с пробелами между тысячами
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', ' ')))
    
    # Удаляем лишние рамки и настраиваем их цвет
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME['neutral'])
    
    # Явно задаем xticks и xticklabels для целых годов
    if hasattr(years, 'min') and hasattr(years, 'max'):
        year_list = list(range(int(years.min()), int(years.max()) + 1))
    else:
        year_list = list(sorted(set(int(y) for y in years)))
    ax.set_xticks(year_list)
    ax.set_xticklabels([str(y) for y in year_list])
    
    return fig

def format_coef(coef):
    """
    Форматирует коэффициент для отображения в уравнении.
    
    Parameters:
    coef (float): Значение коэффициента
    
    Returns:
    str: Отформатированное значение
    """
    if abs(coef) >= 1000000:
        return f"{coef:.2E}".replace('E+0', 'E+')
    elif abs(coef) < 0.001 and coef != 0:
        return f"{coef:.6f}"
    else:
        return f"{coef:.4f}"

def create_coefficients_plot(
    fig: Figure, 
    X: Union[pd.DataFrame, np.ndarray], 
    model: LinearRegression, 
    model_type: str, 
    horizontal: bool = True, 
    max_features: Optional[int] = None
) -> Figure:
    """
    Создает график визуализации коэффициентов модели.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    X: Признаки модели
    model: Обученная модель регрессии
    model_type: Тип модели ('all_groups', 'unemployed', 'combined')
    horizontal: Создавать горизонтальную диаграмму (если False, то вертикальную)
    max_features: Максимальное количество отображаемых признаков
    
    Returns:
    Figure: Обновленный объект фигуры matplotlib
    """
    from ui.components.theme_manager import DARK_THEME
    
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_THEME['bg'])
    
    # Получаем коэффициенты без константы
    coefs = model.coef_
    
    # Цвета для положительных и отрицательных коэффициентов
    pos_color = DARK_THEME['success']
    neg_color = DARK_THEME['error']
    
    # Для модели от возрастных групп и комбинированной модели
    if model_type in ['all_groups', 'combined']:
        # Получаем имена признаков
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f'Признак {i+1}' for i in range(len(coefs))]
        
        # Создаем горизонтальную столбчатую диаграмму для топ коэффициентов
        if max_features is None:
            max_features = 10
            
        if len(coefs) > max_features:
            # Сортируем коэффициенты по абсолютному значению и берем топ
            coef_abs = np.abs(coefs)
            top_indices = np.argsort(coef_abs)[-max_features:]
            top_coefs = coefs[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            y_pos = np.arange(len(top_names))
            if horizontal:
                bars = ax.barh(y_pos, top_coefs, align='center', 
                         color=[pos_color if c > 0 else neg_color for c in top_coefs], alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_names, color=DARK_THEME['neutral'])
            else:
                bars = ax.bar(y_pos, top_coefs, align='center', 
                        color=[pos_color if c > 0 else neg_color for c in top_coefs], alpha=0.8)
                ax.set_xticks(y_pos)
                ax.set_xticklabels(top_names, rotation=45, ha='right', color=DARK_THEME['neutral'])
            
            title = f"Топ-{max_features} коэффициентов по влиянию на ВВП"
        else:
            # Отображаем все коэффициенты
            y_pos = np.arange(len(feature_names))
            if horizontal:
                bars = ax.barh(y_pos, coefs, align='center', 
                         color=[pos_color if c > 0 else neg_color for c in coefs], alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names, color=DARK_THEME['neutral'])
            else:
                bars = ax.bar(y_pos, coefs, align='center', 
                        color=[pos_color if c > 0 else neg_color for c in coefs], alpha=0.8)
                ax.set_xticks(y_pos)
                ax.set_xticklabels(feature_names, rotation=45, ha='right', color=DARK_THEME['neutral'])
            
            title = "Коэффициенты влияния на ВВП"
        
    else:  # model_type == 'unemployed'
        # Для модели от безработицы - простая визуализация
        feature_names = ['Безработица']
        y_pos = np.arange(1)
        if horizontal:
            bars = ax.barh(y_pos, [coefs[0]], align='center', 
                       color=[pos_color if coefs[0] > 0 else neg_color], alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names, color=DARK_THEME['neutral'])
        else:
            bars = ax.bar(y_pos, [coefs[0]], align='center', 
                      color=[pos_color if coefs[0] > 0 else neg_color], alpha=0.8)
            ax.set_xticks(y_pos)
            ax.set_xticklabels(feature_names, color=DARK_THEME['neutral'])
        
        title = "Коэффициент влияния безработицы на ВВП"
    
    if horizontal:
        ax.set_xlabel('Коэффициент', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
        ax.tick_params(axis='x', colors=DARK_THEME['neutral'])
        ax.tick_params(axis='y', colors=DARK_THEME['neutral'])
    else:
        ax.set_ylabel('Коэффициент', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
        ax.tick_params(axis='x', colors=DARK_THEME['neutral'])
        ax.tick_params(axis='y', colors=DARK_THEME['neutral'])
        
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold', color=DARK_THEME['neutral'])
    
    # Добавляем вертикальную/горизонтальную линию в нуле для лучшей видимости
    if horizontal:
        ax.axvline(x=0, color=DARK_THEME['neutral'], linestyle='-', alpha=0.3)
    else:
        ax.axhline(y=0, color=DARK_THEME['neutral'], linestyle='-', alpha=0.3)
    
    # Подписываем значения на столбцах
    for i, bar in enumerate(bars):
        value = bar.get_width() if horizontal else bar.get_height()
        text_color = DARK_THEME['neutral']
        
        if abs(value) > 0.01:
            if abs(value) < 10:
                text = f"{value:.2f}"
            else:
                text = f"{value:.1f}"
            
            # Позиция текста зависит от знака коэффициента и ориентации
            if horizontal:
                if value < 0:
                    ax.text(value - 5, bar.get_y() + bar.get_height()/2, text, 
                            ha='right', va='center', color=text_color, fontweight='bold')
                else:
                    ax.text(value + 5, bar.get_y() + bar.get_height()/2, text, 
                            ha='left', va='center', color=text_color, fontweight='bold')
            else:
                if value < 0:
                    ax.text(bar.get_x() + bar.get_width()/2, value - 5, text, 
                            ha='center', va='top', color=text_color, fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, value + 5, text, 
                            ha='center', va='bottom', color=text_color, fontweight='bold')
    
    # Удаляем лишние рамки и настраиваем их цвет
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME['neutral'])
    
    return fig

def create_residuals_plot(
    fig: Figure,
    years: Union[pd.Series, np.ndarray, range], 
    y: Union[pd.Series, np.ndarray], 
    y_pred: np.ndarray, 
    model_type: str, 
    model_names: Dict[str, str]
) -> Figure:
    """
    Создает график остатков модели.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    years: Значения для оси X (годы или номера наблюдений)
    y: Фактические значения ВВП
    y_pred: Предсказанные значения ВВП
    model_type: Тип модели ('all_groups', 'unemployed', 'combined')
    model_names: Словарь с названиями моделей для заголовков
    
    Returns:
    Figure: Обновленный объект фигуры matplotlib
    """
    ax = fig.add_subplot(111)
    residuals = y - y_pred
    
    # Стандартизированные остатки
    standardized_residuals = residuals / np.std(residuals)
    
    # Точечная диаграмма остатков
    scatter = ax.scatter(y_pred, standardized_residuals, c=standardized_residuals, 
                        cmap='RdYlGn', s=60, alpha=0.8, edgecolor='white')
    ax.axhline(y=0, color='#dc3545', linestyle='-', linewidth=1.5)
    
    # Подписи и оформление
    ax.set_title(f'График стандартизированных остатков (модель от {model_names[model_type]})', 
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Предсказанный ВВП', fontsize=12, labelpad=10)
    ax.set_ylabel('Стандартизированные остатки', fontsize=12, labelpad=10)
    
    # Добавляем горизонтальные линии для значений ±2
    ax.axhline(y=2, color='#28a745', linestyle='--', alpha=0.7)
    ax.axhline(y=-2, color='#28a745', linestyle='--', alpha=0.7)
    
    # Форматирование чисел с пробелами между тысячами
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', ' ')))
    
    # Подписываем годы у точек для лучшей интерпретации
    for i, year in enumerate(years):
        ax.annotate(str(year), (y_pred[i], standardized_residuals[i]), 
                   xytext=(5, 0), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Добавляем цветовую шкалу
    fig.colorbar(scatter, ax=ax)
    
    # Удаляем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def create_dynamics_plot(
    fig: Figure,
    years: Union[pd.Series, np.ndarray, range], 
    y: Union[pd.Series, np.ndarray], 
    X: Union[pd.DataFrame, np.ndarray], 
    model_type: str, 
    model_names: Dict[str, str],
    selected_features: Optional[List[str]] = None
) -> Figure:
    """
    Создает график динамики показателей (нормализованные значения).
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    years: Значения для оси X (годы или номера наблюдений)
    y: Фактические значения ВВП
    X: Признаки модели
    model_type: Тип модели ('all_groups', 'unemployed', 'combined')
    model_names: Словарь с названиями моделей для заголовков
    selected_features: Список выбранных признаков для отображения
    
    Returns:
    Figure: Обновленный объект фигуры matplotlib
    """
    # Импортируем настройки темы
    from ui.components.theme_manager import DARK_THEME
    
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_THEME['bg'])
    
    # Нормализуем ВВП
    gdp_norm = (y - y.min()) / (y.max() - y.min())
    
    # Сначала добавляем ВВП
    ax.plot(years, gdp_norm, marker='o', color=DARK_THEME['neutral'], linewidth=2, markersize=6, label='ВВП (норм.)')
    
    if model_type == 'unemployed':
        # Для модели от безработицы
        if hasattr(X, 'columns'):
            unemployed_col = X.columns[0]
        else:
            unemployed_col = 'Безработица'
            
        unemployed_data = X.iloc[:, 0] if hasattr(X, 'iloc') else X
        unemployed_norm = (unemployed_data - unemployed_data.min()) / (unemployed_data.max() - unemployed_data.min())
        ax.plot(years, unemployed_norm, marker='s', color=DARK_THEME['error'], linewidth=2, markersize=6, label='Безработица (норм.)')
        
        # Рассчитываем корреляцию
        corr = np.corrcoef(gdp_norm, unemployed_norm)[0, 1]
        
        # Добавляем информацию о корреляции
        corr_text = f"Корреляция ВВП и безработицы: {corr:.4f}"
        ax.text(0.02, 0.05, corr_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor=DARK_THEME['bg_light'], alpha=0.8, boxstyle='round,pad=0.5', 
                        edgecolor=DARK_THEME['border']), color=DARK_THEME['neutral'])
        
    elif model_type == 'all_groups' or model_type == 'combined':
        # Для моделей с возрастными группами
        
        # Если указаны конкретные признаки, используем их
        if selected_features is not None and len(selected_features) > 0:
            # Преобразуем имена признаков в индексы, если X - не DataFrame
            if hasattr(X, 'columns'):
                selected_cols = [list(X.columns).index(col) if col in X.columns else i 
                                for i, col in enumerate(selected_features) if i < X.shape[1]]
            else:
                # Если X - не DataFrame, просто берем первые N признаков, где N - длина selected_features
                selected_cols = list(range(min(len(selected_features), X.shape[1])))
        else:
            # Если признаки не указаны, выбираем 4 равномерно распределенных
            if X.shape[1] <= 4:
                selected_cols = list(range(X.shape[1]))
            else:
                # Выбираем 4 равномерно распределенных индекса
                step = max(1, X.shape[1] // 4)
                selected_cols = list(range(0, X.shape[1], step))[:4]
        
        # Используем цвета из темной темы для линий
        colors = [DARK_THEME['accent'], DARK_THEME['success'], 
                 '#fd7e14', DARK_THEME['secondary']]
        
        corr_text = "Корреляции с ВВП:\n"
        
        for i, col_idx in enumerate(selected_cols):
            # Получаем данные и имя признака
            if hasattr(X, 'iloc'):
                feature_data = X.iloc[:, col_idx]
                feature_name = X.columns[col_idx] if hasattr(X, 'columns') else f'Признак {col_idx+1}'
            else:
                feature_data = X[:, col_idx]
                feature_name = f'Признак {col_idx+1}'
            
            # Нормализуем данные
            feature_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            
            # Строим линию
            ax.plot(years, feature_norm, marker='s', color=colors[i % len(colors)], 
                    linewidth=2, markersize=4, label=f'{feature_name}')
            
            # Добавляем корреляцию в текст
            corr = np.corrcoef(gdp_norm, feature_norm)[0, 1]
            corr_text += f"{feature_name}: {corr:.4f}\n"
        
        # Если это комбинированная модель, добавляем безработицу
        if model_type == 'combined':
            # Ищем колонку безработицы
            unemployed_idx = -1
            if hasattr(X, 'columns'):
                for i, col in enumerate(X.columns):
                    if 'безраб' in str(col).lower() or 'unemploy' in str(col).lower():
                        unemployed_idx = i
                        break
            
            if unemployed_idx != -1:
                unemployed_data = X.iloc[:, unemployed_idx]
                unemployed_norm = (unemployed_data - unemployed_data.min()) / (unemployed_data.max() - unemployed_data.min())
                ax.plot(years, unemployed_norm, marker='^', color=DARK_THEME['error'], linewidth=2, markersize=6, label='Безработица (норм.)')
                
                # Добавляем корреляцию в текст
                corr = np.corrcoef(gdp_norm, unemployed_norm)[0, 1]
                corr_text += f"Безработица: {corr:.4f}"
        
        # Добавляем информацию о корреляциях с нужными цветами из темной темы
        ax.text(0.02, 0.02, corr_text, transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor=DARK_THEME['bg_light'], alpha=0.9, boxstyle='round,pad=0.5', 
                        edgecolor=DARK_THEME['border']), color=DARK_THEME['neutral'])
    
    ax.set_title(f'Динамика ВВП и {model_names[model_type]} (нормализованные значения)', 
                 fontsize=14, pad=20, fontweight='bold', color=DARK_THEME['neutral'])
    ax.set_xlabel('Год', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
    ax.set_ylabel('Нормализованное значение', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
    
    # Настраиваем легенду под темную тему
    legend = ax.legend(loc='best', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor(DARK_THEME['bg_light'])
    legend.get_frame().set_edgecolor(DARK_THEME['border'])
    
    # Обновляем цвет текста в легенде
    for text in legend.get_texts():
        text.set_color(DARK_THEME['neutral'])
    
    # Настраиваем цвет делений на осях
    ax.tick_params(axis='x', rotation=45, colors=DARK_THEME['neutral'])
    ax.tick_params(axis='y', colors=DARK_THEME['neutral'])
    
    # Настраиваем сетку
    ax.grid(True, alpha=0.5, color=DARK_THEME['bg_light'], linestyle='--')
    
    # Удаляем лишние рамки и настраиваем их цвет
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME['neutral'])
    
    # Явно задаем xticks и xticklabels для целых годов
    if hasattr(years, 'min') and hasattr(years, 'max'):
        year_list = list(range(int(years.min()), int(years.max()) + 1))
    else:
        year_list = list(sorted(set(int(y) for y in years)))
    ax.set_xticks(year_list)
    ax.set_xticklabels([str(y) for y in year_list])
    
    return fig

def export_all_plots(
    df: pd.DataFrame, 
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    model: LinearRegression, 
    y_pred: np.ndarray, 
    model_type: str, 
    file_path: str
) -> bool:
    """
    Экспортирует все графики в PDF-файл.
    
    Parameters:
    df (pandas.DataFrame): Датафрейм с данными
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (LinearRegression): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    model_type (str): Тип модели ('all_groups', 'unemployed', 'combined')
    file_path (str): Путь к файлу для сохранения
    
    Returns:
    bool: True если экспорт успешен, False в случае ошибки
    """
    try:
        logger.info(f"Экспорт графиков в PDF файл: {file_path}")
        from matplotlib.backends.backend_pdf import PdfPages
        from tkinter import messagebox
        
        with PdfPages(file_path) as pdf:
            # Экспортируем все типы графиков
            for i in range(4):  # 4 типа графиков
                fig = create_graph(i, df, X, y, model, y_pred, model_type)
                pdf.savefig(fig)
                plt.close(fig)
            
        # Показываем сообщение об успешном экспорте
        messagebox.showinfo("Экспорт графиков", f"Все графики экспортированы в PDF:\n{file_path}")
        logger.info(f"Графики успешно экспортированы в {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте графиков: {str(e)}")
        messagebox.showerror("Ошибка экспорта", f"Не удалось экспортировать графики:\n{str(e)}")
        return False

def get_graph_title(graph_index: int, model_type: str, model_names: Dict[str, str]) -> str:
    """
    Возвращает заголовок для графика по его индексу и типу модели.
    
    Parameters:
    graph_index (int): Индекс типа графика
    model_type (str): Тип модели ('all_groups', 'unemployed', 'combined')
    model_names (Dict[str, str]): Словарь с названиями моделей
    
    Returns:
    str: Заголовок графика
    """
    titles = {
        0: f"Фактический и прогнозируемый ВВП (модель от {model_names[model_type]})",
        1: "Коэффициенты влияния на ВВП",
        2: f"График стандартизированных остатков (модель от {model_names[model_type]})",
        3: f"Динамика ВВП и {model_names[model_type]} (нормализованные значения)"
    }
    
    return titles.get(graph_index, f"График {graph_index+1}")

def save_graph_to_file(
    fig: Figure, 
    file_path: str, 
    dpi: int = 300, 
    bbox_inches: str = 'tight'
) -> bool:
    """
    Сохраняет график в файл указанного формата.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    file_path (str): Путь к файлу для сохранения
    dpi (int): Разрешение изображения
    bbox_inches (str): Параметр обрезки границ
    
    Returns:
    bool: True если сохранение успешно, False в случае ошибки
    """
    try:
        logger.info(f"Сохранение графика в файл: {file_path}")
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        logger.info(f"График успешно сохранен в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении графика: {str(e)}")
        return False

def resize_graph(fig: Figure, width: int, height: int) -> Figure:
    """
    Изменяет размер графика.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    width (int): Новая ширина в дюймах
    height (int): Новая высота в дюймах
    
    Returns:
    Figure: Объект фигуры с новым размером
    """
    fig.set_size_inches(width, height)
    fig.tight_layout()
    return fig

def apply_style_to_graph(fig: Figure, style_name: str = 'dark') -> Figure:
    """
    Применяет указанный стиль к графику.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    style_name (str): Название стиля ('dark', 'light', 'default')
    
    Returns:
    Figure: Объект фигуры с примененным стилем
    """
    # Импортируем настройки темы
    from ui.components.theme_manager import DARK_THEME
    
    # Настройка цвета фона фигуры
    fig.patch.set_facecolor(DARK_THEME['primary'])
    
    # Применяем стиль к осям
    for ax in fig.get_axes():
        # Настройка цвета фона оси
        ax.set_facecolor(DARK_THEME['bg'])
        
        # Настройка цвета текста и меток
        ax.xaxis.label.set_color(DARK_THEME['neutral'])
        ax.yaxis.label.set_color(DARK_THEME['neutral'])
        ax.title.set_color(DARK_THEME['neutral'])
        
        # Настройка цвета делений и линий сетки
        ax.tick_params(colors=DARK_THEME['neutral'])
        ax.grid(True, alpha=0.5, color=DARK_THEME['bg_light'], linestyle='--')
        
        # Настройка цвета рамки
        for spine in ax.spines.values():
            spine.set_color(DARK_THEME['neutral'])
        
        # Обновляем цвет текста в легенде, если она есть
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(DARK_THEME['bg_light'])
            legend.get_frame().set_edgecolor(DARK_THEME['border'])
            
            for text in legend.get_texts():
                text.set_color(DARK_THEME['neutral'])
    
    # Настраиваем стиль конкретных типов графиков
    if style_name == 'dark':
        # Темный режим (используем DARK_THEME)
        pass
    elif style_name == 'light':
        # Если нужно будет поддерживать светлую тему, здесь можно добавить соответствующие настройки
        pass
    elif style_name == 'default':
        # Настройки по умолчанию
        pass
    
    return fig

def add_watermark(fig: Figure, text: str = "Регрессионный анализ") -> Figure:
    """
    Добавляет водяной знак на график.
    
    Parameters:
    fig (Figure): Объект фигуры matplotlib
    text (str): Текст водяного знака
    
    Returns:
    Figure: Объект фигуры с водяным знаком
    """
    # Создаем подграфик для водяного знака, который будет перекрывать основной
    ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax.text(0.5, 0.5, text, fontsize=40, color='gray', 
           alpha=0.2, ha='center', va='center', rotation=30)
    ax.axis('off')
    
    return fig