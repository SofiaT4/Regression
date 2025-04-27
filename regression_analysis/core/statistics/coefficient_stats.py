"""
Модуль для статистического анализа коэффициентов регрессионных моделей.

Содержит функции для расчета статистических показателей отдельных коэффициентов 
модели, включая p-значения, t-статистики, доверительные интервалы и стандартные ошибки.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Any, Union, Optional

from .regression_stats import calculate_statistics

def calculate_coefficient_pvalues(coefficients: np.ndarray, 
                                 se_coefficients: np.ndarray, 
                                 df_residual: int) -> List[float]:
    """
    Рассчитывает p-значения для коэффициентов регрессии.
    
    Parameters:
    coefficients (numpy.ndarray): Коэффициенты модели
    se_coefficients (numpy.ndarray): Стандартные ошибки коэффициентов
    df_residual (int): Степени свободы остатков
    
    Returns:
    List[float]: p-значения для коэффициентов
    """
    if df_residual <= 0:
        return [1.0] * len(coefficients)
    
    # Расчет t-статистики
    t_values = coefficients / se_coefficients
    
    # Расчет p-значений (двусторонний тест)
    p_values = [2 * (1 - stats.t.cdf(abs(t), df_residual)) for t in t_values]
    
    return p_values

def calculate_confidence_intervals(coefficients: np.ndarray, 
                                  se_coefficients: np.ndarray, 
                                  df_residual: int, 
                                  confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Рассчитывает доверительные интервалы для коэффициентов.
    
    Parameters:
    coefficients (numpy.ndarray): Коэффициенты модели
    se_coefficients (numpy.ndarray): Стандартные ошибки коэффициентов
    df_residual (int): Степени свободы остатков
    confidence (float): Уровень доверия (по умолчанию 0.95)
    
    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]: Нижние и верхние границы доверительных интервалов
    """
    if df_residual <= 0:
        return coefficients.copy(), coefficients.copy()
    
    # Критическое значение t для заданного уровня доверия
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df_residual)
    
    # Расчет доверительных интервалов
    lower_ci = coefficients - t_critical * se_coefficients
    upper_ci = coefficients + t_critical * se_coefficients
    
    return lower_ci, upper_ci

def identify_significant_coefficients(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Определяет статистически значимые коэффициенты.
    
    Parameters:
    p_values (List[float]): p-значения коэффициентов
    alpha (float): Уровень значимости (по умолчанию 0.05)
    
    Returns:
    List[bool]: Список булевых значений, True для значимых коэффициентов
    """
    return [p < alpha for p in p_values]

def calculate_standardized_coefficients(X: Union[pd.DataFrame, np.ndarray], 
                                       coefficients: np.ndarray) -> np.ndarray:
    """
    Рассчитывает стандартизированные коэффициенты регрессии (бета-коэффициенты).
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    coefficients (numpy.ndarray): Коэффициенты модели (без константы)
    
    Returns:
    numpy.ndarray: Стандартизированные коэффициенты
    """
    # Если X это DataFrame, преобразуем в numpy array
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    # Стандартные отклонения признаков
    X_std = np.std(X_array, axis=0, ddof=1)
    
    # Стандартизированные коэффициенты
    standardized_coeffs = coefficients[1:] * (X_std / np.std(coefficients[0] + X_array.dot(coefficients[1:]), ddof=1))
    
    # Добавляем 0 для константы
    return np.hstack([[0], standardized_coeffs])

def calculate_anova_table(X: Union[pd.DataFrame, np.ndarray], 
                         y: Union[pd.Series, np.ndarray], 
                         model: Any,
                         y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Рассчитывает таблицу дисперсионного анализа (ANOVA) для регрессионной модели.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (object): Обученная модель регрессии
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
    
    # Создаем словарь с результатами дисперсионного анализа
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
        'p_value_f': p_value_f
    }
    
    return anova_results

def calculate_regression_diagnostics(X: Union[pd.DataFrame, np.ndarray], 
                                  y: Union[pd.Series, np.ndarray], 
                                  model: Any,
                                  y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Рассчитывает дополнительные диагностические показатели регрессии.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): Признаки модели
    y (pandas.Series or numpy.ndarray): Целевая переменная
    model (object): Обученная модель регрессии
    y_pred (numpy.ndarray): Предсказанные значения
    
    Returns:
    Dict[str, Any]: Словарь с диагностическими показателями регрессии
    """
    # Преобразуем Series в numpy массивы для однородности
    if isinstance(y, pd.Series):
        y = y.values
    
    # Удаляем NaN значения для расчета
    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Остатки
    residuals = y_valid - y_pred_valid
    
    # Среднее и стандартное отклонение остатков
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals, ddof=1)
    
    # Стандартизированные остатки
    standardized_residuals = residuals / residuals_std
    
    # Значения теоретических квантилей для нормального распределения
    n = len(residuals)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    
    # Тест Дарбина-Уотсона на автокорреляцию остатков
    # Значения близкие к 2 означают отсутствие автокорреляции,
    # < 1 или > 3 - положительная или отрицательная автокорреляция
    dw_statistic = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    
    # Тест Жарке-Бера на нормальность остатков
    # Низкое p-значение означает отклонение от нормальности
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    jb_statistic = n / 6 * (skewness ** 2 + kurtosis ** 2 / 4)
    jb_pvalue = 1 - stats.chi2.cdf(jb_statistic, 2)
    
    # Создаем словарь с диагностическими показателями
    diagnostics = {
        'residuals_mean': residuals_mean,
        'residuals_std': residuals_std,
        'standardized_residuals': standardized_residuals,
        'theoretical_quantiles': theoretical_quantiles,
        'durbin_watson': dw_statistic,
        'jarque_bera_statistic': jb_statistic,
        'jarque_bera_pvalue': jb_pvalue,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
    
    return diagnostics

def calculate_vif(X: pd.DataFrame) -> Dict[str, float]:
    """
    Рассчитывает фактор инфляции дисперсии (VIF) для выявления мультиколлинеарности.
    
    Parameters:
    X (pandas.DataFrame): Признаки модели
    
    Returns:
    Dict[str, float]: Словарь с VIF для каждого признака
    """
    from sklearn.linear_model import LinearRegression
    
    vif = {}
    
    # VIF рассчитывается для каждого признака отдельно
    for i, col in enumerate(X.columns):
        # Выделяем один признак как зависимую переменную
        y = X[col]
        
        # Остальные признаки - независимые переменные
        X_others = X.drop(col, axis=1)
        
        # Строим модель, где один признак зависит от остальных
        if X_others.shape[1] > 0:  # Проверка, что есть другие признаки
            model = LinearRegression()
            model.fit(X_others, y)
            
            # Получаем предсказания
            y_pred = model.predict(X_others)
            
            # Рассчитываем R^2
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            # VIF = 1 / (1 - R^2)
            if r2 < 1:  # Проверка, чтобы избежать деления на 0
                vif[col] = 1 / (1 - r2)
            else:
                vif[col] = float('inf')  # Бесконечность в случае высокой коллинеарности
        else:
            vif[col] = 1.0  # Если это единственный признак, VIF = 1
    
    return vif

def calculate_all_statistics(X_all_groups: Union[pd.DataFrame, np.ndarray], 
                            X_unemployed: Union[pd.DataFrame, np.ndarray], 
                            X_combined: Union[pd.DataFrame, np.ndarray], 
                            y: Union[pd.Series, np.ndarray], 
                            models: Dict[str, LinearRegression], 
                            predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """
    Рассчитывает статистические показатели для всех трех моделей.
    Включает полный набор статистик для дисперсионного анализа в формате Excel.
    
    Parameters:
    X_all_groups (pandas.DataFrame or numpy.ndarray): Признаки - все возрастные группы
    X_unemployed (pandas.DataFrame or numpy.ndarray): Признаки - безработица
    X_combined (pandas.DataFrame or numpy.ndarray): Признаки - все возрастные группы и безработица
    y (pandas.Series or numpy.ndarray): Целевая переменная - ВВП
    models (dict): Словарь с обученными моделями
    predictions (dict): Словарь с предсказаниями моделей
    
    Returns:
    dict: Словарь со статистическими показателями для всех трех моделей
    """
    # Получаем базовые статистики для всех моделей
    stats_dict = {
        'all_groups': calculate_statistics(X_all_groups, y, models['all_groups'], predictions['all_groups']),
        'unemployed': calculate_statistics(X_unemployed, y, models['unemployed'], predictions['unemployed']),
        'combined': calculate_statistics(X_combined, y, models['combined'], predictions['combined'])
    }
    
    # Добавляем расчет дисперсионного анализа для каждой модели
    anova_all_groups = calculate_anova_table(X_all_groups, y, models['all_groups'], predictions['all_groups'])
    anova_unemployed = calculate_anova_table(X_unemployed, y, models['unemployed'], predictions['unemployed'])
    anova_combined = calculate_anova_table(X_combined, y, models['combined'], predictions['combined'])
    
    # Объединяем результаты дисперсионного анализа с остальными статистиками
    for key, value in anova_all_groups.items():
        stats_dict['all_groups'][key] = value
    
    for key, value in anova_unemployed.items():
        stats_dict['unemployed'][key] = value
        
    for key, value in anova_combined.items():
        stats_dict['combined'][key] = value
    
    # Дополнительный расчет для всех моделей в формате Excel
    for model_type in stats_dict:
        X = {'all_groups': X_all_groups, 'unemployed': X_unemployed, 'combined': X_combined}[model_type]
        coefficients = stats_dict[model_type]['coefficients']
        
        # Получаем имена признаков в формате как в Excel
        if model_type == 'all_groups':
            # Создаем имена в формате X1, X2, X3 и т.д.
            feature_names = ['Y-пересечение']
            if hasattr(X, 'columns'):
                for i, col in enumerate(X.columns):
                    feature_names.append(f'Переменная X {i+1}')
            else:
                for i in range(X.shape[1]):
                    feature_names.append(f'Переменная X {i+1}')
                    
            stats_dict[model_type]['excel_feature_names'] = feature_names
            
        elif model_type == 'unemployed':
            # Для модели безработицы
            feature_names = ['Y-пересечение', 'Переменная X 1']
            stats_dict[model_type]['excel_feature_names'] = feature_names
            
        else:  # combined
            # Для комбинированной модели
            feature_names = ['Y-пересечение']
            if hasattr(X, 'columns'):
                for i, col in enumerate(X.columns):
                    feature_names.append(f'Переменная X {i+1}')
            else:
                for i in range(X.shape[1]):
                    feature_names.append(f'Переменная X {i+1}')
                    
            stats_dict[model_type]['excel_feature_names'] = feature_names
        
        try:
            # Добавляем расчет VIF для каждой модели, если X является DataFrame
            if hasattr(X, 'columns'):
                vif_values = calculate_vif(X)
                # Добавляем VIF = 0 для константы (Y-пересечения)
                vif_list = [0.0]  # Константа не имеет VIF
                for col in X.columns:
                    vif_list.append(vif_values.get(col, 1.0))
                stats_dict[model_type]['vif'] = vif_list
                
                # Добавляем максимальное значение VIF как отдельный показатель
                if len(vif_values) > 0:
                    stats_dict[model_type]['max_vif'] = max(vif_values.values())
                else:
                    stats_dict[model_type]['max_vif'] = 1.0
                
            # Рассчитываем стандартизированные коэффициенты, если возможно
            std_coeffs = calculate_standardized_coefficients(X, coefficients)
            stats_dict[model_type]['standardized_coefficients'] = std_coeffs
            
            # Добавляем 95.0% доверительные интервалы (аналогичные 95%)
            if 'lower_ci' in stats_dict[model_type] and 'upper_ci' in stats_dict[model_type]:
                stats_dict[model_type]['lower_ci_95_0'] = stats_dict[model_type]['lower_ci']
                stats_dict[model_type]['upper_ci_95_0'] = stats_dict[model_type]['upper_ci']
                
            # Добавляем диагностические показатели для регрессии
            diagnostics = calculate_regression_diagnostics(X, y, models[model_type], predictions[model_type])
            for key, value in diagnostics.items():
                stats_dict[model_type][key] = value
                
            # Добавляем дополнительные статистические показатели для моделей
            stats_dict[model_type]['dw_statistic'] = diagnostics.get('durbin_watson', 0)
            stats_dict[model_type]['jb_statistic'] = diagnostics.get('jarque_bera_statistic', 0)
            stats_dict[model_type]['jb_pvalue'] = diagnostics.get('jarque_bera_pvalue', 1.0)
            stats_dict[model_type]['skewness'] = diagnostics.get('skewness', 0)
            stats_dict[model_type]['kurtosis'] = diagnostics.get('kurtosis', 0)
            
            # Добавляем обобщающие показатели: сумма квадратов ошибок (SSE)
            # и среднеквадратическая ошибка (RMSE)
            residuals = y - predictions[model_type]
            valid_residuals = residuals[~np.isnan(residuals)]
            stats_dict[model_type]['sse'] = np.sum(valid_residuals**2)
            stats_dict[model_type]['rmse'] = np.sqrt(np.mean(valid_residuals**2))
            
            # Добавляем информационные критерии (AIC и BIC)
            n = len(y[~np.isnan(y)])
            k = X.shape[1] + 1  # количество параметров (включая константу)
            sse = stats_dict[model_type]['sse']
            
            if n > k and sse > 0:
                # Расчет AIC и BIC
                aic = n * np.log(sse/n) + 2*k
                bic = n * np.log(sse/n) + k*np.log(n)
                
                stats_dict[model_type]['aic'] = aic
                stats_dict[model_type]['bic'] = bic
            else:
                stats_dict[model_type]['aic'] = float('inf')
                stats_dict[model_type]['bic'] = float('inf')
                
        except Exception as e:
            print(f"Ошибка при расчете дополнительных статистик для модели {model_type}: {e}")
            # Если не удалось рассчитать, заполняем стандартными значениями
            stats_dict[model_type]['standardized_coefficients'] = np.zeros_like(coefficients)
            stats_dict[model_type]['max_vif'] = 1.0
            stats_dict[model_type]['dw_statistic'] = 2.0
            stats_dict[model_type]['jb_statistic'] = 0.0
            stats_dict[model_type]['jb_pvalue'] = 1.0
            stats_dict[model_type]['sse'] = 0.0
            stats_dict[model_type]['rmse'] = 0.0
            stats_dict[model_type]['aic'] = float('inf')
            stats_dict[model_type]['bic'] = float('inf')
            
            if 'lower_ci' in stats_dict[model_type] and 'upper_ci' in stats_dict[model_type]:
                stats_dict[model_type]['lower_ci_95_0'] = stats_dict[model_type]['lower_ci']
                stats_dict[model_type]['upper_ci_95_0'] = stats_dict[model_type]['upper_ci']
    
    return stats_dict

def get_most_significant_coefficients(model_stats: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, float, float]]:
    """
    Возвращает наиболее значимые коэффициенты модели.
    
    Parameters:
    model_stats (Dict[str, Any]): Статистические показатели модели
    top_n (int): Количество коэффициентов для возврата
    
    Returns:
    List[Tuple[str, float, float]]: Список кортежей (имя признака, коэффициент, p-значение)
    """
    coefficients = model_stats['coefficients']
    p_values = model_stats['p_values']
    
    # Используем Excel-форматированные имена, если доступны
    if 'excel_feature_names' in model_stats:
        feature_names = model_stats['excel_feature_names']
    else:
        feature_names = model_stats['feature_names']
    
    # Создаем список кортежей (имя, коэффициент, p-значение)
    coef_data = [(feature_names[i], coefficients[i], p_values[i]) for i in range(len(coefficients))]
    
    # Сортируем по p-значению (по возрастанию) и берем top_n
    coef_data.sort(key=lambda x: x[2])
    return coef_data[:top_n]

def compare_models_statistics(model_stats1: Dict[str, Any], 
                             model_stats2: Dict[str, Any]) -> Dict[str, float]:
    """
    Сравнивает статистические показатели двух моделей.
    
    Parameters:
    model_stats1 (Dict[str, Any]): Статистические показатели первой модели
    model_stats2 (Dict[str, Any]): Статистические показатели второй модели
    
    Returns:
    Dict[str, float]: Словарь с относительными изменениями показателей
    """
    comparison = {}
    
    # Сравниваем основные показатели
    for metric in ['r2', 'adjusted_r2', 'se_regression', 'f_statistic']:
        if metric in model_stats1 and metric in model_stats2 and model_stats1[metric] != 0:
            comparison[metric] = (model_stats2[metric] - model_stats1[metric]) / model_stats1[metric]
    
    return comparison