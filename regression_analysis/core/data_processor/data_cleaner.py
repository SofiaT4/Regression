"""
Модуль для очистки и предобработки данных в приложении регрессионного анализа.

Содержит функции для обработки пропущенных значений, нормализации данных,
преобразования типов данных и подготовки данных к регрессионному анализу.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List, Optional, Union

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и подготавливает данные для анализа.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    
    Returns:
    pandas.DataFrame: Очищенный датафрейм
    """
    # Удаляем пустые строки и столбцы
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Преобразуем данные в числовой формат
    for col in df.columns:
        try:
            # Преобразуем только нечисловые колонки, кроме "Год"
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Заменяем запятые на точки для десятичных чисел
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                
                # Пробуем преобразовать в числовой формат
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Ошибка при преобразовании колонки '{col}': {str(e)}")
    
    # Проверяем, что год в числовом формате
    if 'Год' in df.columns and not pd.api.types.is_numeric_dtype(df['Год']):
        try:
            df['Год'] = pd.to_numeric(df['Год'], errors='coerce')
        except:
            # Если не удалось преобразовать, заменяем на последовательность
            df['Год'] = range(2000, 2000 + len(df))
    
    # Удаляем строки с пропущенными значениями в обязательных колонках
    required_columns = ['Год', 'ВВП (в текущих ценах)']
    if all(col in df.columns for col in required_columns):
        df = df.dropna(subset=required_columns)
    
    # Сортируем по годам, если такая колонка есть
    if 'Год' in df.columns:
        df = df.sort_values('Год')
    
    return df

def handle_missing_values(df: pd.DataFrame, method: str = 'drop', 
                         fill_value: Optional[Union[str, float, Dict]] = None) -> pd.DataFrame:
    """
    Обрабатывает пропущенные значения в датафрейме.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    method (str): Метод обработки пропусков ('drop', 'mean', 'median', 'zero', 'value')
    fill_value: Значение для заполнения пропусков при method='value'
                или словарь {column: value} для заполнения по колонкам
    
    Returns:
    pandas.DataFrame: Датафрейм с обработанными пропущенными значениями
    """
    df_copy = df.copy()
    
    if method == 'drop':
        # Удаляем строки с пропущенными значениями
        df_copy = df_copy.dropna()
    
    elif method == 'mean':
        # Заполняем пропуски средними значениями по колонкам
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    elif method == 'median':
        # Заполняем пропуски медианными значениями по колонкам
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    elif method == 'zero':
        # Заполняем пропуски нулями
        df_copy = df_copy.fillna(0)
    
    elif method == 'value' and fill_value is not None:
        # Заполняем пропуски заданным значением
        if isinstance(fill_value, dict):
            # Если передан словарь, заполняем по колонкам
            for col, val in fill_value.items():
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].fillna(val)
        else:
            # Если передано одно значение, заполняем им все пропуски
            df_copy = df_copy.fillna(fill_value)
    
    return df_copy

def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                      decimal: str = '.', thousands: str = ' ') -> pd.DataFrame:
    """
    Преобразует указанные колонки в числовой формат.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    columns (List[str], optional): Список колонок для преобразования (если None, преобразуются все колонки)
    decimal (str): Десятичный разделитель
    thousands (str): Разделитель тысяч
    
    Returns:
    pandas.DataFrame: Датафрейм с преобразованными колонками
    """
    df_copy = df.copy()
    
    cols_to_convert = columns if columns is not None else df_copy.columns
    
    for col in cols_to_convert:
        if col in df_copy.columns:
            try:
                # Если колонка уже в числовом формате, пропускаем
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    continue
                
                # Преобразуем в строки для обработки
                series = df_copy[col].astype(str)
                
                # Заменяем разделители в соответствии с указанными параметрами
                if decimal != '.':
                    series = series.str.replace(decimal, '.')
                
                if thousands != '':
                    series = series.str.replace(thousands, '')
                
                # Удаляем другие нечисловые символы (кроме точки и знака минус)
                series = series.str.replace(r'[^\d.-]', '', regex=True)
                
                # Преобразуем в числовой формат
                df_copy[col] = pd.to_numeric(series, errors='coerce')
                
            except Exception as e:
                print(f"Ошибка при преобразовании колонки '{col}' в числовой формат: {str(e)}")
    
    return df_copy

def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 method: str = 'minmax') -> pd.DataFrame:
    """
    Нормализует числовые данные в датафрейме.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    columns (List[str], optional): Список колонок для нормализации (если None, нормализуются все числовые колонки)
    method (str): Метод нормализации ('minmax', 'zscore')
    
    Returns:
    pandas.DataFrame: Датафрейм с нормализованными данными
    """
    df_copy = df.copy()
    
    # Если не указаны колонки, нормализуем все числовые
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            if method == 'minmax':
                # Min-Max нормализация в диапазон [0, 1]
                min_val = df_copy[col].min()
                max_val = df_copy[col].max()
                
                # Проверяем, чтобы не было деления на ноль
                if max_val > min_val:
                    df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
                else:
                    # Если все значения одинаковые, устанавливаем 0.5
                    df_copy[col] = 0.5
            
            elif method == 'zscore':
                # Z-score нормализация (среднее = 0, стд = 1)
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                
                # Проверяем, чтобы не было деления на ноль
                if std > 0:
                    df_copy[col] = (df_copy[col] - mean) / std
                else:
                    # Если стандартное отклонение = 0, устанавливаем 0
                    df_copy[col] = 0
    
    return df_copy