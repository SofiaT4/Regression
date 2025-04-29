"""
Модуль для построения различных регрессионных моделей.

Содержит функции для создания, обучения и оценки моделей линейной регрессии,
а также обработки ошибок при построении моделей.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from typing import Dict, Tuple, Optional, List, Any, Union

def check_multicollinearity(X):
    """
    Проверяет наличие мультиколлинеарности между признаками.
    
    Parameters:
    X (pandas.DataFrame): Признаки модели
    
    Returns:
    List: Список пар признаков с высокой корреляцией
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)
    
    # Check for high correlations
    high_corr_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[0]):
            if abs(corr_matrix[i, j]) > 0.8:  # Threshold for high correlation
                high_corr_pairs.append((X.columns[i], X.columns[j], corr_matrix[i, j]))
    
    return high_corr_pairs

def build_regression_model(X: pd.DataFrame, y: pd.Series) -> Tuple[LinearRegression, np.ndarray]:
    """
    Строит и обучает модель линейной регрессии.
    
    Parameters:
    X (pandas.DataFrame): Features
    y (pandas.Series): Target variable
    
    Returns:
    Tuple[LinearRegression, np.ndarray]: Обученная модель и прогнозируемые значения
    """
    
    if X.empty or len(X.columns) == 0:
        raise ValueError("Получен пустой датафрейм признаков")
    
    # Получаем строки без пропущенных значений как в признаках, так и в целевой переменной
    valid_rows = X.dropna().index.intersection(y.dropna().index)
    if len(valid_rows) == 0:
        raise ValueError("Нет данных без пропущенных значений для обучения модели")
    
    # Создаем копии данных для обучения, чтобы избежать предупреждений о изменении исходных данных
    X_clean = X.loc[valid_rows].copy()
    y_clean = y.loc[valid_rows].copy()
    
    # Важно! Выводим информацию о признаках для проверки
    print(f"Обучение модели с {X_clean.shape[1]} признаками: {X_clean.columns.tolist()}")
    
    # Проверка на мультиколлинеарность
    if X_clean.shape[1] > 1:
        corr_matrix = X_clean.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
        
        if high_corr_cols:
            print(f"Предупреждение: Обнаружена сильная корреляция между признаками: {high_corr_cols}")
            # Мы только предупреждаем, но не удаляем колонки, чтобы не нарушать структуру данных
    
    # ИСПРАВЛЕНИЕ: Удаляем или модифицируем логику выбора признаков на основе корреляции
    # Это может быть причиной изменения количества переменных
    
    # Проверка соотношения количества наблюдений и признаков
    if len(X_clean) <= X_clean.shape[1]:
        print("Предупреждение: количество наблюдений меньше или равно количеству признаков.")
        print("Это может привести к переобучению модели.")
        # НЕ выбираем признаки, а используем регуляризацию для предотвращения переобучения
        model = Ridge(alpha=0.1)  # Используем Ridge вместо линейной регрессии
    else:
        # Обучаем обычную модель линейной регрессии
        model = LinearRegression()
    
    # Обучаем модель, используя ВСЕ доступные признаки
    model.fit(X_clean, y_clean)
    
    print(f"Коэффициенты модели: {model.coef_}")
    print(f"Константа модели: {model.intercept_}")
    
    # ИСПРАВЛЕНИЕ: Проверяем, что количество коэффициентов совпадает с количеством признаков
    if hasattr(model, 'coef_') and len(model.coef_) != X_clean.shape[1]:
        print(f"ОШИБКА: Количество коэффициентов ({len(model.coef_)}) не совпадает с количеством признаков ({X_clean.shape[1]})")
    
    # Создаем массив для предсказаний, изначально заполненный NaN
    y_pred = np.full_like(y.values, np.nan, dtype=float)
    
    # ИСПРАВЛЕНИЕ: Убеждаемся, что X_for_prediction содержит только те же столбцы, что и X_clean
    X_for_prediction = X[X_clean.columns].copy()
    
    # Делаем предсказания только для строк без пропущенных значений
    pred_valid_rows = X_for_prediction.dropna().index
    
    if len(pred_valid_rows) > 0:
        X_pred = X_for_prediction.loc[pred_valid_rows]
        y_pred_valid = model.predict(X_pred)
        
        # Заполняем массив предсказаний
        for i, idx in enumerate(pred_valid_rows):
            if idx < len(y_pred):
                y_pred[idx] = y_pred_valid[i]
    
    # Заполняем пропущенные предсказания средним значением
    valid_preds = ~np.isnan(y_pred)
    if np.any(valid_preds):
        mean_pred = np.mean(y_pred[valid_preds])
        y_pred[~valid_preds] = mean_pred
    else:
        # Если все предсказания NaN, используем среднее значение y
        y_pred[:] = y.mean()
    
    return model, y_pred

def predict_with_model(model: LinearRegression, X: pd.DataFrame, 
                     original_X: Optional[pd.DataFrame] = None, 
                     original_y: Optional[pd.Series] = None) -> np.ndarray:
    """
    Делает предсказания с использованием обученной модели.
    
    Parameters:
    model (LinearRegression): Обученная модель регрессии
    X (pandas.DataFrame): Данные для предсказания
    original_X (pandas.DataFrame, optional): Исходные данные, на которых обучалась модель
    original_y (pandas.Series, optional): Исходные целевые значения
    
    Returns:
    np.ndarray: Предсказанные значения
    """
    # Если исходных данных нет, делаем простое предсказание
    if original_X is None or original_y is None:
        try:
            return model.predict(X)
        except Exception as e:
            print(f"Ошибка при предсказании: {str(e)}")
            return np.full(X.shape[0], np.nan)
    
    # Делаем предсказание с учетом структуры исходных данных
    y_pred = np.zeros_like(original_y, dtype=float)
    
    # Используем только те строки, для которых у нас есть все признаки без пропусков
    valid_rows = X.dropna().index
    
    try:
        # Если модель обучалась на подмножестве признаков, корректируем данные для предсказания
        if hasattr(model, 'feature_names_in_'):
            # Если scikit-learn >= 1.0
            model_features = model.feature_names_in_
            X_for_prediction = X.loc[valid_rows, model_features]
        else:
            # Для более старых версий scikit-learn или если признаки не сохранены
            X_for_prediction = X.loc[valid_rows]
        
        # Предсказываем значения для валидных строк
        y_pred_valid = model.predict(X_for_prediction)
        y_pred[valid_rows] = y_pred_valid
    except Exception as e:
        print(f"Ошибка при предсказании: {str(e)}")
        # В случае ошибки заполняем средними значениями
        y_pred[valid_rows] = original_y.loc[valid_rows].mean()
    
    return y_pred

def handle_model_errors(model: Optional[LinearRegression], X: pd.DataFrame, y: pd.Series) -> Tuple[LinearRegression, np.ndarray]:
    """
    Обрабатывает ошибки при построении модели и создает запасную модель при необходимости.
    
    Parameters:
    model (LinearRegression, optional): Модель, которая может быть None в случае ошибки
    X (pandas.DataFrame): Признаки модели
    y (pandas.Series): Целевая переменная
    
    Returns:
    Tuple[LinearRegression, np.ndarray]: Модель (возможно фиктивную) и предсказания
    """
    if model is None:
        # Создаем фиктивную модель
        dummy_model = LinearRegression()
        dummy_model.coef_ = np.zeros(1 if X.shape[1] == 0 else X.shape[1])
        dummy_model.intercept_ = y.mean()
        
        # Создаем фиктивные предсказания
        y_pred = np.full_like(y, y.mean())
        
        return dummy_model, y_pred
    
    return model, predict_with_model(model, X, X, y)

def build_models(X_all_groups: pd.DataFrame, X_unemployed: pd.DataFrame, 
                 X_combined: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, LinearRegression], Dict[str, np.ndarray]]:
    """
    Построение трех различных моделей линейной регрессии.
    
    Parameters:
    X_all_groups (pandas.DataFrame): Признаки - все возрастные группы
    X_unemployed (pandas.DataFrame): Признаки - безработица
    X_combined (pandas.DataFrame): Признаки - все возрастные группы и безработица
    y (pandas.Series): Целевая переменная - ВВП
    
    Returns:
    Tuple[Dict[str, LinearRegression], Dict[str, np.ndarray]]: 
        Словари с обученными моделями и их предсказаниями
    """
    # Создаем и обучаем модели линейной регрессии с обработкой ошибок
    models = {}
    predictions = {}
    
    # 1. Модель по возрастным группам
    try:
        if X_all_groups.shape[1] > 0:  # Проверяем, что есть колонки для обучения
            model_all_groups, y_pred_all_groups = build_regression_model(X_all_groups, y)
            models['all_groups'] = model_all_groups
            predictions['all_groups'] = y_pred_all_groups
        else:
            raise ValueError("Недостаточно признаков для обучения модели all_groups")
    except Exception as e:
        print(f"Ошибка при построении модели all_groups: {str(e)}")
        # Обрабатываем ошибку и создаем фиктивную модель
        dummy_model, y_pred = handle_model_errors(None, X_all_groups, y)
        models['all_groups'] = dummy_model
        predictions['all_groups'] = y_pred
    
    # 2. Модель по безработице
    try:
        if X_unemployed.shape[1] > 0:
            model_unemployed, y_pred_unemployed = build_regression_model(X_unemployed, y)
            models['unemployed'] = model_unemployed
            predictions['unemployed'] = y_pred_unemployed
        else:
            raise ValueError("Недостаточно признаков для обучения модели unemployed")
    except Exception as e:
        print(f"Ошибка при построении модели unemployed: {str(e)}")
        # Обрабатываем ошибку и создаем фиктивную модель
        dummy_model, y_pred = handle_model_errors(None, X_unemployed, y)
        models['unemployed'] = dummy_model
        predictions['unemployed'] = y_pred
    
    # 3. Комбинированная модель
    try:
        if X_combined.shape[1] > 0:
            # Добавляем диагностическую информацию
            print(f"X_combined shape: {X_combined.shape}")
            print(f"X_combined columns: {X_combined.columns.tolist() if hasattr(X_combined, 'columns') else 'No columns'}")
            print(f"y shape: {y.shape}")
            
            # Проверяем мультиколлинеарность
            high_corr = check_multicollinearity(X_combined)
            if high_corr:
                print(f"Warning: High multicollinearity detected between: {high_corr}")
                # Можно добавить код для удаления высококоррелирующих признаков здесь
                # Например, оставлять только один из каждой пары
                
            model_combined, y_pred_combined = build_regression_model(X_combined, y)
            models['combined'] = model_combined
            predictions['combined'] = y_pred_combined

            # Добавьте для отладки:
            print(f"Combined model coefficients: {model_combined.coef_}")
            print(f"Combined model intercept: {model_combined.intercept_}")
            print(f"Combined model prediction range: {np.min(y_pred_combined)} - {np.max(y_pred_combined)}")
        else:
            raise ValueError("Недостаточно признаков для обучения модели combined")
    except Exception as e:
        print(f"Ошибка при построении модели combined: {str(e)}")
        # Обрабатываем ошибку и создаем фиктивную модель
        dummy_model, y_pred = handle_model_errors(None, X_combined, y)
        models['combined'] = dummy_model
        predictions['combined'] = y_pred
    
    return models, predictions