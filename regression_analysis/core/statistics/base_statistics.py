"""
Модуль с базовыми статистическими функциями для регрессионного анализа.

Содержит основные статистические расчеты и обработку случаев с 
ограниченным количеством наблюдений для регрессионного анализа.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Dict, Any, Union, List, Optional, Tuple

def limited_statistics(X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray], 
                      model: Any, 
                      y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Расчет ограниченной статистики, когда число наблюдений недостаточно для полного анализа.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (object): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    dict: Словарь с базовой статистикой
    """
    n = len(y)  # количество наблюдений
    k = X.shape[1]  # количество независимых переменных (без константы)
    
    # Базовые показатели, которые можно вычислить без достаточного количества наблюдений
    r2 = r2_score(y, y_pred) if n > 1 else 1.0
    multiple_r = np.sqrt(r2)
    
    # Попытка вычислить adjusted R2, если возможно
    if n > k + 1:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    else:
        adjusted_r2 = r2  # Просто возвращаем R2, если формула не применима
    
    # Коэффициенты модели без расчета статистической значимости
    if hasattr(model, 'intercept_'):
        intercept = model.intercept_
    else:
        intercept = 0.0
        
    if hasattr(model, 'coef_'):
        coefficients = np.hstack([[intercept], model.coef_])
    else:
        coefficients = np.array([intercept] + [0.0] * k)
    
    # Заполняем пустыми значениями статистики, которые не можем вычислить
    se_coefficients = np.zeros_like(coefficients)
    t_values = np.zeros_like(coefficients)
    p_values = [1.0] * len(coefficients)
    lower_ci = coefficients.copy()
    upper_ci = coefficients.copy()
    
    # Формируем имена признаков
    if hasattr(X, 'columns'):
        feature_names = ['Константа'] + list(X.columns)
    else:
        feature_names = ['Константа'] + [f'Признак {i+1}' for i in range(k)]
    
    # Возвращаем словарь с ограниченной статистикой
    return {
        'multiple_r': multiple_r,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'se_regression': 0.0,
        'observations': n,
        'df_regression': k,
        'df_residual': max(0, n - k - 1),
        'df_total': max(0, n - 1),
        'ss_regression': 0.0,
        'ss_residual': 0.0,
        'ss_total': 0.0,
        'ms_regression': 0.0,
        'ms_residual': 0.0,
        'f_statistic': 0.0,
        'p_value_f': 1.0,
        'coefficients': coefficients,
        'se_coefficients': se_coefficients,
        't_values': t_values,
        'p_values': p_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'feature_names': feature_names
    }

def calculate_sums_of_squares(y: Union[pd.Series, np.ndarray], 
                             y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Рассчитывает суммы квадратов для регрессионного анализа.
    
    Parameters:
    y (pandas.Series or numpy.ndarray): Фактические значения
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    Tuple[float, float, float]: (ss_total, ss_regression, ss_residual)
    """
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)  # Общая сумма квадратов
    ss_residual = np.sum((y - y_pred) ** 2)  # Сумма квадратов остатков
    ss_regression = ss_total - ss_residual  # Сумма квадратов регрессии
    
    return ss_total, ss_regression, ss_residual

def calculate_degrees_of_freedom(n: int, k: int) -> Tuple[int, int, int]:
    """
    Рассчитывает степени свободы для регрессионного анализа.
    
    Parameters:
    n (int): Количество наблюдений
    k (int): Количество независимых переменных (без константы)
    
    Returns:
    Tuple[int, int, int]: (df_total, df_regression, df_residual)
    """
    df_total = n - 1  # Общие степени свободы
    df_regression = k  # Степени свободы регрессии
    df_residual = n - k - 1  # Степени свободы остатков
    
    return df_total, df_regression, df_residual

def calculate_mean_squares(ss_regression: float, ss_residual: float, 
                          df_regression: int, df_residual: int) -> Tuple[float, float]:
    """
    Рассчитывает средние квадраты для регрессионного анализа.
    
    Parameters:
    ss_regression (float): Сумма квадратов регрессии
    ss_residual (float): Сумма квадратов остатков
    df_regression (int): Степени свободы регрессии
    df_residual (int): Степени свободы остатков
    
    Returns:
    Tuple[float, float]: (ms_regression, ms_residual)
    """
    ms_regression = ss_regression / df_regression if df_regression > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    
    return ms_regression, ms_residual

def check_sample_size(n: int, k: int) -> bool:
    """
    Проверяет, достаточно ли наблюдений для полного регрессионного анализа.
    
    Parameters:
    n (int): Количество наблюдений
    k (int): Количество независимых переменных (без константы)
    
    Returns:
    bool: True, если наблюдений достаточно, False в противном случае
    """
    # Для полноценного регрессионного анализа нужно, чтобы число наблюдений
    # превышало число переменных + 1 (константа)
    return n > k + 1