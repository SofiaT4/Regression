"""
Модуль для расчета статистических показателей регрессионных моделей.

Содержит функции для расчета R², скорректированного R², 
стандартной ошибки, F-статистики и других показателей качества 
регрессионных моделей в формате, совместимом с Excel.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from typing import Dict, Any, Union, List, Optional, Tuple

from core.statistics.base_statistics import (
    limited_statistics,
    calculate_sums_of_squares,
    calculate_degrees_of_freedom,
    calculate_mean_squares,
    check_sample_size
)

def calculate_r_squared(y: Union[pd.Series, np.ndarray], 
                       y_pred: np.ndarray) -> float:
    """
    Расчет коэффициента детерминации (R-квадрат).
    
    Parameters:
    y (pandas.Series or numpy.ndarray): Фактические значения
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    float: Коэффициент детерминации
    """
    return r2_score(y, y_pred)

def calculate_adjusted_r_squared(r2: float, n: int, k: int) -> float:
    """
    Расчет скорректированного коэффициента детерминации.
    
    Parameters:
    r2 (float): Коэффициент детерминации
    n (int): Число наблюдений
    k (int): Число независимых переменных (без константы)
    
    Returns:
    float: Скорректированный коэффициент детерминации
    """
    if n <= k + 1:
        return r2  # Если недостаточно наблюдений, возвращаем обычный R2
    
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def calculate_standard_error(y: Union[pd.Series, np.ndarray], 
                           y_pred: np.ndarray, 
                           df_residual: int) -> float:
    """
    Расчет стандартной ошибки регрессии.
    
    Parameters:
    y (pandas.Series or numpy.ndarray): Фактические значения
    y_pred (numpy.ndarray): Предсказанные значения
    df_residual (int): Степени свободы остатков
    
    Returns:
    float: Стандартная ошибка регрессии
    """
    if df_residual <= 0:
        return 0.0
    
    mse = mean_squared_error(y, y_pred)
    se = np.sqrt(mse * len(y) / df_residual)
    return se

def calculate_f_statistic(ms_regression: float, ms_residual: float) -> float:
    """
    Расчет F-статистики для модели.
    
    Parameters:
    ms_regression (float): Средний квадрат регрессии
    ms_residual (float): Средний квадрат остатков
    
    Returns:
    float: F-статистика
    """
    if ms_residual <= 0:
        return 0.0
    
    return ms_regression / ms_residual

def calculate_f_pvalue(f_statistic: float, df_regression: int, df_residual: int) -> float:
    """
    Расчет p-значения для F-статистики.
    
    Parameters:
    f_statistic (float): F-статистика
    df_regression (int): Степени свободы регрессии
    df_residual (int): Степени свободы остатков
    
    Returns:
    float: p-значение для F-статистики
    """
    if df_regression <= 0 or df_residual <= 0 or f_statistic <= 0:
        return 1.0
    
    try:
        p_value = stats.f.sf(f_statistic, df_regression, df_residual)
        return p_value
    except Exception as e:
        print(f"Ошибка при расчете p-значения F-статистики: {e}")
        return 1.0

def calculate_anova_table(X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray],
                        y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Рассчитывает таблицу дисперсионного анализа (ANOVA) для регрессионной модели.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    Dict[str, Any]: Словарь с показателями дисперсионного анализа
    """
    # Преобразуем Series в numpy массивы для однородности
    if isinstance(y, pd.Series):
        y = y.values
    
    # Удаляем NaN значения для расчета
    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Количество наблюдений и признаков
    n = len(y_valid)  # количество наблюдений
    k = X.shape[1]  # количество независимых переменных (без константы)
    
    # Расчет сумм квадратов
    y_mean = np.mean(y_valid)
    ss_total = np.sum((y_valid - y_mean) ** 2)  # Общая сумма квадратов
    ss_residual = np.sum((y_valid - y_pred_valid) ** 2)  # Сумма квадратов остатков
    ss_regression = ss_total - ss_residual  # Сумма квадратов регрессии
    
    # Расчет степеней свободы
    df_regression = k  # Степени свободы регрессии
    df_residual = n - k - 1  # Степени свободы остатков
    df_total = n - 1  # Общие степени свободы
    
    # Расчет средних квадратов
    ms_regression = ss_regression / df_regression if df_regression > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    
    # Расчет F-статистики и p-значения
    f_statistic = ms_regression / ms_residual if ms_residual > 0 else 0
    
    try:
        p_value_f = stats.f.sf(f_statistic, df_regression, df_residual)
    except:
        p_value_f = 1.0
    
    # Формируем результаты ANOVA в формате как в Excel
    anova_results = {
        'df_regression': df_regression,
        'df_residual': df_residual,
        'df_total': df_total,
        'ss_regression': ss_regression,
        'ss_residual': ss_residual,
        'ss_total': ss_total,
        'ms_regression': ms_regression,
        'ms_residual': ms_residual,
        'f_statistic': f_statistic,
        'p_value_f': p_value_f,
        'anova_table': {
            'regression': {
                'df': df_regression,
                'ss': ss_regression,
                'ms': ms_regression,
                'f': f_statistic,
                'p_value': p_value_f
            },
            'residual': {
                'df': df_residual,
                'ss': ss_residual,
                'ms': ms_residual
            },
            'total': {
                'df': df_total,
                'ss': ss_total
            }
        }
    }
    
    return anova_results

def calculate_excel_statistics(X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              model: Any,
                              y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Расчет статистических показателей модели в формате Excel.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (object): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    dict: Словарь всех статистических показателей в формате Excel
    """
    # Преобразуем Series в numpy массивы для однородности
    if isinstance(y, pd.Series):
        y = y.values
    
    # Удаляем NaN значения для расчета
    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    n = len(y_valid)  # количество наблюдений
    k = X.shape[1]  # количество независимых переменных (без константы)
    
    # Вычисляем показатели дисперсионного анализа
    anova_results = calculate_anova_table(X, y_valid, y_pred_valid)
    
    # Извлекаем основные показатели из результатов ANOVA
    ss_total = anova_results['ss_total']
    ss_residual = anova_results['ss_residual']
    ss_regression = anova_results['ss_regression']
    df_regression = anova_results['df_regression']
    df_residual = anova_results['df_residual']
    df_total = anova_results['df_total']
    ms_regression = anova_results['ms_regression']
    ms_residual = anova_results['ms_residual']
    f_statistic = anova_results['f_statistic']
    p_value_f = anova_results['p_value_f']
    
    # R-квадрат и множественный R
    if ss_total == 0:
        r2 = 0
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    multiple_r = np.sqrt(max(0, r2))
    
    # Нормированный R-квадрат (скорректированный)
    if n <= k + 1:
        adjusted_r2 = r2
    else:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    # Стандартная ошибка модели
    se_regression = np.sqrt(ms_residual) if ms_residual > 0 else 0
    
    # Готовим данные в точно таком же формате, как в Excel
    excel_stats = {
        'multiple_r': multiple_r,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'se_regression': se_regression,
        'observations': n,
        'df_regression': df_regression,
        'df_residual': df_residual,
        'df_total': df_total,
        'ss_regression': ss_regression,
        'ss_residual': ss_residual,
        'ss_total': ss_total,
        'ms_regression': ms_regression,
        'ms_residual': ms_residual,
        'f_statistic': f_statistic,
        'p_value_f': p_value_f,
        'anova_table': anova_results['anova_table']
    }
    
    # Добавляем коэффициенты и их статистики
    if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
        # Коэффициенты модели (включая константу)
        coefficients = np.hstack([[model.intercept_], model.coef_])
        
        # Формируем матрицу X для расчета статистик коэффициентов
        if isinstance(X, pd.DataFrame):
            X_with_const = pd.DataFrame({'const': np.ones(len(X))}, index=X.index)
            for col in X.columns:
                X_with_const[col] = X[col]
            X_with_const = X_with_const.values
        else:
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])
        
        # Фильтруем только валидные строки
        X_with_const_valid = X_with_const[valid_mask]
        
        # Расчет ковариационной матрицы коэффициентов
        try:
            X_transpose_X = X_with_const_valid.T.dot(X_with_const_valid)
            X_transpose_X_inv = np.linalg.inv(X_transpose_X)
            cov_matrix = X_transpose_X_inv * ms_residual
            se_coefficients = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            print("Предупреждение: обнаружена мультиколлинеарность. Используется псевдообратная матрица.")
            X_transpose_X_pinv = np.linalg.pinv(X_with_const_valid.T.dot(X_with_const_valid))
            cov_matrix = X_transpose_X_pinv * ms_residual
            se_coefficients = np.sqrt(np.diag(cov_matrix))
        
        # Расчет t-статистики и p-значений
        t_values = coefficients / se_coefficients
        p_values = [2 * (1 - stats.t.cdf(abs(t), df_residual)) for t in t_values]
        
        # Доверительные интервалы для коэффициентов (95%)
        t_critical = stats.t.ppf(0.975, df_residual)
        lower_ci = coefficients - t_critical * se_coefficients
        upper_ci = coefficients + t_critical * se_coefficients
        
        # Формируем имена признаков как в Excel
        feature_names = ['Y-пересечение']
        if hasattr(X, 'columns'):
            for i, col in enumerate(X.columns):
                feature_names.append(f'Переменная X{i+1}')
        else:
            for i in range(X.shape[1]):
                feature_names.append(f'Переменная X{i+1}')
        
        # Добавляем в выходной словарь
        excel_stats.update({
            'coefficients': coefficients,
            'se_coefficients': se_coefficients,
            't_values': t_values,
            'p_values': p_values,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'feature_names': feature_names,
            'excel_feature_names': feature_names
        })
        
        # Добавляем словарь с коэффициентами в формате как в Excel
        excel_stats['coefficients_table'] = []
        for i, name in enumerate(feature_names):
            coef_row = {
                'name': name,
                'coefficient': coefficients[i],
                'std_error': se_coefficients[i],
                't_stat': t_values[i],
                'p_value': p_values[i],
                'lower_95': lower_ci[i],
                'upper_95': upper_ci[i],
                'lower_95_0': lower_ci[i],  # Дублируем для совместимости с Excel
                'upper_95_0': upper_ci[i]   # Дублируем для совместимости с Excel
            }
            excel_stats['coefficients_table'].append(coef_row)
    
    return excel_stats

def calculate_statistics(X: Union[pd.DataFrame, np.ndarray], 
                        y: Union[pd.Series, np.ndarray], 
                        model: Any, 
                        y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Расчет статистических показателей модели точно как в Excel (Анализ данных -> Регрессия).
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (object): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    dict: Словарь статистических показателей
    """
    # Преобразуем Series в numpy массивы для однородности
    if isinstance(y, pd.Series):
        y = y.values
    
    # Удаляем NaN значения для расчета
    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_valid) == 0:
        # Возвращаем ограниченную статистику, если нет валидных данных
        return limited_statistics(X, y, model, y_pred)
    
    n = len(y_valid)  # количество наблюдений
    k = X.shape[1]  # количество независимых переменных (без константы)
    
    # Проверяем на случай слишком малого количества наблюдений
    if not check_sample_size(n, k):
        # Ограниченная статистика при недостаточном числе наблюдений
        return limited_statistics(X, y, model, y_pred)
    
    # Используем функцию для расчета статистики в формате Excel
    return calculate_excel_statistics(X, y, model, y_pred)

def format_anova_table_for_display(anova_table: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Форматирует таблицу дисперсионного анализа для отображения в интерфейсе.
    
    Parameters:
    anova_table (Dict[str, Dict[str, Any]]): Таблица дисперсионного анализа
    
    Returns:
    List[Dict[str, str]]: Список строк с отформатированными значениями
    """
    formatted_rows = []
    
    # Строка "Регрессия"
    regression_row = {
        'source': 'Регрессия',
        'df': str(anova_table['regression']['df']),
        'ss': f"{anova_table['regression']['ss']:.8f}",
        'ms': f"{anova_table['regression']['ms']:.8f}",
        'f': f"{anova_table['regression']['f']:.8f}",
        'p_value': f"{anova_table['regression']['p_value']:.8f}"
    }
    formatted_rows.append(regression_row)
    
    # Строка "Остаток"
    residual_row = {
        'source': 'Остаток',
        'df': str(anova_table['residual']['df']),
        'ss': f"{anova_table['residual']['ss']:.8f}",
        'ms': f"{anova_table['residual']['ms']:.8f}",
        'f': '',
        'p_value': ''
    }
    formatted_rows.append(residual_row)
    
    # Строка "Итого"
    total_row = {
        'source': 'Итого',
        'df': str(anova_table['total']['df']),
        'ss': f"{anova_table['total']['ss']:.8f}",
        'ms': '',
        'f': '',
        'p_value': ''
    }
    formatted_rows.append(total_row)
    
    return formatted_rows

def format_statistics_for_display(stats: Dict[str, Any]) -> Dict[str, str]:
    """
    Форматирует статистические показатели для отображения в интерфейсе.
    
    Parameters:
    stats (Dict[str, Any]): Словарь статистических показателей
    
    Returns:
    Dict[str, str]: Словарь отформатированных показателей
    """
    formatted_stats = {}
    
    # Форматируем основные показатели
    formatted_stats['multiple_r'] = f"{stats['multiple_r']:.8f}"
    formatted_stats['r2'] = f"{stats['r2']:.8f}"
    formatted_stats['adjusted_r2'] = f"{stats['adjusted_r2']:.8f}"
    formatted_stats['se_regression'] = f"{stats['se_regression']:.8f}"
    formatted_stats['observations'] = str(stats['observations'])
    
    # Форматируем показатели дисперсионного анализа
    formatted_stats['df_regression'] = str(stats['df_regression'])
    formatted_stats['df_residual'] = str(stats['df_residual'])
    formatted_stats['df_total'] = str(stats['df_total'])
    
    formatted_stats['ss_regression'] = f"{stats['ss_regression']:.8f}"
    formatted_stats['ss_residual'] = f"{stats['ss_residual']:.8f}"
    formatted_stats['ss_total'] = f"{stats['ss_total']:.8f}"
    
    formatted_stats['ms_regression'] = f"{stats['ms_regression']:.8f}"
    formatted_stats['ms_residual'] = f"{stats['ms_residual']:.8f}"
    
    formatted_stats['f_statistic'] = f"{stats['f_statistic']:.8f}"
    formatted_stats['p_value_f'] = f"{stats['p_value_f']:.8f}"
    
    # Добавляем оценку модели
    if stats['p_value_f'] < 0.05:
        formatted_stats['model_significance'] = "Статистически значима (p < 0.05)"
    else:
        formatted_stats['model_significance'] = "Статистически незначима (p ≥ 0.05)"
    
    if stats['r2'] > 0.7:
        formatted_stats['model_quality'] = "Высокое качество модели (R² > 0.7)"
    elif stats['r2'] > 0.5:
        formatted_stats['model_quality'] = "Среднее качество модели (0.5 < R² < 0.7)"
    else:
        formatted_stats['model_quality'] = "Низкое качество модели (R² < 0.5)"
    
    # Формируем таблицу дисперсионного анализа в человекочитаемом формате
    if 'anova_table' in stats:
        formatted_stats['anova_table'] = format_anova_table_for_display(stats['anova_table'])
    
    return formatted_stats

def create_anova_summary(stats: Dict[str, Any]) -> str:
    """
    Создает текстовое резюме результатов дисперсионного анализа.
    
    Parameters:
    stats (Dict[str, Any]): Статистические показатели модели
    
    Returns:
    str: Текстовое резюме дисперсионного анализа
    """
    f_value = stats['f_statistic']
    p_value = stats['p_value_f']
    r2 = stats['r2']
    adj_r2 = stats['adjusted_r2']
    
    summary = "Результаты дисперсионного анализа (ANOVA):\n\n"
    
    if p_value < 0.001:
        significance = "высоко статистически значима"
    elif p_value < 0.01:
        significance = "статистически значима"
    elif p_value < 0.05:
        significance = "статистически значима на стандартном уровне"
    else:
        significance = "статистически незначима"
    
    summary += f"Модель регрессии {significance} (F = {f_value:.4f}, p = {p_value:.6f}).\n"
    
    if r2 > 0.8:
        quality = "очень высокое"
    elif r2 > 0.6:
        quality = "высокое"
    elif r2 > 0.4:
        quality = "среднее"
    elif r2 > 0.2:
        quality = "низкое"
    else:
        quality = "очень низкое"
    
    summary += f"Качество модели {quality} (R² = {r2:.4f}, скорректированный R² = {adj_r2:.4f}).\n"
    summary += f"Модель объясняет {r2*100:.1f}% дисперсии зависимой переменной.\n"
    
    return summary