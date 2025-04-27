"""
Модуль для выбора и обработки признаков в регрессионном анализе.

Содержит функции для определения возрастных групп, отбора признаков 
для моделей регрессии и поиска признаков с высокой корреляцией.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Optional, Dict, Any

def detect_age_groups(df: pd.DataFrame) -> List[str]:
    """
    Автоматическое определение колонок с возрастными группами.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    
    Returns:
    List[str]: Список названий колонок с возрастными группами
    """
    age_groups = []
    
    # Ищем колонки, содержащие возрастные диапазоны в формате XX-XX
    age_pattern = re.compile(r'\d+\s*-\s*\d+')
    
    for col in df.columns:
        col_str = str(col)
        
        # Игнорируем колонки, которые очевидно не являются возрастными группами
        if col_str in ['Год', 'ВВП (в текущих ценах)', 'Численность безработных в возрасте 15-72 лет (Тыс. человек)']:
            continue
            
        # Проверяем, содержит ли название колонки возрастной диапазон
        if age_pattern.search(col_str):
            age_groups.append(col)
            continue
            
        # Проверяем другие ключевые слова, указывающие на возрастные группы
        col_lower = col_str.lower()
        if (('возраст' in col_lower or 'лет' in col_lower) and 
            not 'безраб' in col_lower):
            age_groups.append(col)
    
    # Если паттерн не найден по названию колонки, проверяем данные в колонке
    # (для случаев, когда возрастные группы - это просто числа в заголовках)
    if not age_groups:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Год', 'ВВП (в текущих ценах)', 'Численность безработных в возрасте 15-72 лет (Тыс. человек)']:
                # Проверяем, что в колонке нет отрицательных значений (характерно для возрастов)
                if df[col].min() >= 0:
                    age_groups.append(col)
    
    return age_groups

def find_correlated_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> List[str]:
    """
    Определяет признаки с высокой корреляцией с целевой переменной.
    
    Parameters:
    X (pandas.DataFrame): Признаки
    y (pandas.Series): Целевая переменная
    threshold (float): Порог корреляции для отбора признаков
    
    Returns:
    List[str]: Список названий признаков с корреляцией выше порога
    """
    correlated_features = []
    
    # Проверяем корреляцию каждого признака с целевой переменной
    for col in X.columns:
        try:
            corr = X[col].corr(y)
            if abs(corr) >= threshold:
                correlated_features.append(col)
        except:
            # Пропускаем признаки, для которых не удается вычислить корреляцию
            pass
    
    return correlated_features

def select_features(X: pd.DataFrame, y: pd.Series, max_features: Optional[int] = None) -> pd.DataFrame:
    """
    Выбирает наиболее важные признаки для модели.
    
    Parameters:
    X (pandas.DataFrame): Исходные признаки
    y (pandas.Series): Целевая переменная
    max_features (int, optional): Максимальное количество признаков для отбора
    
    Returns:
    pandas.DataFrame: Датафрейм с отобранными признаками
    """
    # Если число признаков не превышает лимит или лимит не задан, возвращаем все признаки
    if max_features is None or X.shape[1] <= max_features:
        return X
    
    # Если число наблюдений меньше числа признаков, уменьшаем размерность
    if len(X) <= X.shape[1]:
        # Выбираем наиболее коррелирующие с целевой переменной признаки
        correlations = []
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                correlations.append((col, abs(corr)))
            except:
                correlations.append((col, 0))
        
        # Сортируем по абсолютному значению корреляции
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Выбираем топ-N признаков, где N < число наблюдений
        top_features = [corr[0] for corr in correlations[:min(max_features, len(X)-1)]]
        return X[top_features]
    
    # В противном случае выбираем топ-N по корреляции
    correlations = []
    for col in X.columns:
        try:
            corr = X[col].corr(y)
            correlations.append((col, abs(corr)))
        except:
            correlations.append((col, 0))
    
    # Сортируем по абсолютному значению корреляции
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Выбираем топ-N признаков
    top_features = [corr[0] for corr in correlations[:max_features]]
    return X[top_features]

def handle_feature_selection_for_regression(X: pd.DataFrame, y: pd.Series, min_observations_per_feature: int = 5) -> pd.DataFrame:
    """
    Обрабатывает выбор признаков для регрессионного анализа с учетом размера выборки.
    
    Parameters:
    X (pandas.DataFrame): Признаки модели
    y (pandas.Series): Целевая переменная
    min_observations_per_feature (int): Минимальное количество наблюдений на один признак
    
    Returns:
    pandas.DataFrame: Отобранные признаки для модели
    """
    # Проверяем размер выборки относительно количества признаков
    n_observations = len(X)
    n_features = X.shape[1]
    
    # Расчёт максимального количества признаков исходя из размера выборки
    max_allowed_features = n_observations // min_observations_per_feature
    
    # Если признаков слишком много, выбираем только самые важные
    if n_features > max_allowed_features and max_allowed_features > 0:
        print(f"Предупреждение: слишком много признаков ({n_features}) для размера выборки ({n_observations}).")
        print(f"Выбираем только {max_allowed_features} наиболее коррелирующих признаков.")
        
        return select_features(X, y, max_allowed_features)
    
    return X