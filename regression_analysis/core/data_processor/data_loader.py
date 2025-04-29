"""
Модуль для загрузки и первичной обработки данных в приложении регрессионного анализа.

Содержит функции для чтения CSV файлов, определения кодировок и разделителей,
а также подготовки данных для регрессионного анализа.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional, Dict, Any

def detect_encoding_and_delimiter(file_path: str) -> Tuple[str, str]:
    """
    Определяет кодировку и разделитель CSV файла.
    
    Parameters:
    file_path (str): Путь к CSV файлу
    
    Returns:
    Tuple[str, str]: Кодировка и разделитель файла
    """
    # Список возможных кодировок для проверки
    encodings = ['utf-8', 'cp1251', 'latin1', 'ascii', 'cp1252']
    file_content = None
    detected_encoding = None
    
    # Пытаемся определить кодировку
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                file_content = f.read(1000)  # Читаем первые 1000 символов
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    
    # Если не удалось определить кодировку, используем utf-8 и читаем с обработкой ошибок
    if detected_encoding is None:
        detected_encoding = 'utf-8'
        with open(file_path, 'rb') as f:
            file_content = f.read(1000).decode('utf-8', errors='replace')
    
    # Определяем сепаратор по содержимому
    detected_delimiter = ','
    if file_content:
        # Проверяем наличие разделителей в файле
        separators = {
            ',': file_content.count(','),
            ';': file_content.count(';'),
            '\t': file_content.count('\t')
        }
        
        # Выбираем наиболее часто встречающийся разделитель
        most_common_sep = max(separators.items(), key=lambda x: x[1])[0]
        if separators[most_common_sep] > 0:
            detected_delimiter = most_common_sep
    
    return detected_encoding, detected_delimiter

def read_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Читает CSV файл с определённой кодировкой и разделителем.
    
    Parameters:
    file_path (str): Путь к CSV файлу
    **kwargs: Дополнительные параметры для pandas.read_csv
    
    Returns:
    pd.DataFrame: Загруженный датафрейм
    """
    # Определяем кодировку и разделитель, если не указаны
    encoding = kwargs.pop('encoding', None)
    delimiter = kwargs.pop('sep', None)
    
    if encoding is None or delimiter is None:
        auto_encoding, auto_delimiter = detect_encoding_and_delimiter(file_path)
        if encoding is None:
            encoding = auto_encoding
        if delimiter is None:
            delimiter = auto_delimiter
    
    # Пробуем различные комбинации параметров для чтения файла
    success = False
    error_message = ""
    
    # Список сепараторов для проверки
    separators_to_try = [delimiter, ';', ',', '\t']
    
    for sep in separators_to_try:
        try:
            # Пробуем прочитать файл
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=sep,
                **kwargs
            )
            
            # Проверяем, что прочитано больше одной колонки
            if df.shape[1] > 1:
                # Если всё хорошо, возвращаем датафрейм
                return df
        except Exception as e:
            error_message = str(e)
    
    # Если не удалось прочитать файл ни с одним разделителем
    raise ValueError(f"Не удалось прочитать CSV-файл: {error_message}")

def prepare_data(df: pd.DataFrame, age_groups: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Подготовка данных для трех различных регрессионных моделей.
    
    Parameters:
    df (pandas.DataFrame): Исходный датафрейм с данными
    age_groups (list): Список колонок с возрастными группами (опционально)
    
    Returns:
    tuple: (X_all_groups, X_unemployed, X_combined, y) - наборы признаков для трех моделей и целевая переменная
    """
    # Делаем копию, чтобы не изменять оригинал
    df = df.copy()
    
    # Проверяем наличие необходимых колонок
    required_cols = {
        'Год': ['год', 'year'],
        'ВВП (в текущих ценах)': ['ввп', 'gdp', 'врп', 'валов', 'продукт'],
        'Численность безработных в возрасте 15-72 лет (Тыс. человек)': ['безраб', 'unemploy']
    }
    
    # Проверяем, существуют ли необходимые колонки, и если нет, ищем подходящие
    for req_col, keywords in required_cols.items():
        if req_col not in df.columns:
            found = False
            for col in df.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in keywords):
                    df = df.rename(columns={col: req_col})
                    found = True
                    break
            
            # Если не нашли подходящую колонку, создаем ее
            if not found:
                if req_col == 'Год':
                    df['Год'] = range(2000, 2000 + len(df))
                    print("Колонка 'Год' не найдена. Создана последовательность лет, начиная с 2000.")
                elif req_col == 'ВВП (в текущих ценах)':
                    # Ищем любую числовую колонку, которая еще не использована
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols:
                            if col != 'Год' and col != 'Численность безработных в возрасте 15-72 лет (Тыс. человек)':
                                df['ВВП (в текущих ценах)'] = df[col]
                                print(f"Колонка '{col}' используется как 'ВВП'.")
                                break
                    
                    # Если не нашли подходящую числовую колонку, создаем случайные данные
                    if 'ВВП (в текущих ценах)' not in df.columns:
                        df['ВВП (в текущих ценах)'] = np.random.randint(1000000, 10000000, len(df))
                        print("Колонка 'ВВП' не найдена. Созданы случайные данные для демонстрации.")
                elif req_col == 'Численность безработных в возрасте 15-72 лет (Тыс. человек)':
                    df[req_col] = np.random.randint(3000, 6000, len(df))
                    print("Колонка 'Безработные' не найдена. Созданы случайные данные для демонстрации.")
    
    # Если возрастные группы не переданы, пытаемся определить их автоматически
    if age_groups is None or len(age_groups) == 0:
        # Получаем колонки с возрастными группами
        age_groups = []
        
        # Ищем колонки, содержащие возрастные диапазоны в формате XX-XX
        age_pattern = re.compile(r'\d+\s*-\s*\d+')
        
        for col in df.columns:
            col_str = str(col)
            
            # Игнорируем обязательные колонки
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
        
        # Печатаем информацию о найденных возрастных группах
        if age_groups:
            print(f"Обнаружено {len(age_groups)} возрастных групп: {age_groups}")
        else:
            print("Возрастные группы не обнаружены.")
    
    # 1. Модель: ВВП от всех возрастных групп
    if age_groups and len(age_groups) > 0:
        # Проверяем наличие колонок возрастных групп в датафрейме
        valid_age_groups = [col for col in age_groups if col in df.columns]
        
        if valid_age_groups:
            X_all_groups = df[valid_age_groups].copy()
            print(f"Модель 'all_groups' использует {len(valid_age_groups)} признаков: {valid_age_groups}")
        else:
            # Если не найдены указанные возрастные группы, создаем пустой датафрейм с одной колонкой
            X_all_groups = pd.DataFrame({
                'Численность рабочих (итого)': np.random.randint(10000, 50000, len(df))
            }, index=df.index)
            print("Указанные возрастные группы не найдены в данных. Создана фиктивная колонка 'Численность рабочих (итого)'.")
    else:
        # Если возрастные группы не найдены, создаем пустой датафрейм с одной колонкой
        X_all_groups = pd.DataFrame({
            'Численность рабочих (итого)': np.random.randint(10000, 50000, len(df))
        }, index=df.index)
        print("Возрастные группы не найдены. Создана фиктивная колонка 'Численность рабочих (итого)'.")
    
    # 2. Модель: ВВП от безработицы
    # Явно создаем DataFrame с одним столбцом
    unemployed_col = 'Численность безработных в возрасте 15-72 лет (Тыс. человек)'
    X_unemployed = pd.DataFrame({
        unemployed_col: df[unemployed_col].values
    }, index=df.index)
    print(f"Модель 'unemployed' использует колонку '{unemployed_col}'")
    
    # 3. Модель: ВВП от всех возрастных групп и безработицы
    if len(X_all_groups.columns) > 0:
        # Создаем копию DataFrame для возрастных групп
        X_combined = X_all_groups.copy()
        # Добавляем колонку с безработицей
        X_combined[unemployed_col] = df[unemployed_col].values
        print(f"Модель 'combined' использует {len(X_combined.columns)} признаков: {list(X_combined.columns)}")
    else:
        X_combined = pd.concat([X_all_groups, X_unemployed], axis=1)
        print(f"Модель 'combined' использует фиктивные данные и безработицу: {list(X_combined.columns)}")
    
    # Целевая переменная для всех моделей
    y = df['ВВП (в текущих ценах)']
    
    # Проверяем, что все DataFrame созданы корректно
    assert isinstance(X_all_groups, pd.DataFrame), "X_all_groups должен быть DataFrame"
    assert isinstance(X_unemployed, pd.DataFrame), "X_unemployed должен быть DataFrame"
    assert isinstance(X_combined, pd.DataFrame), "X_combined должен быть DataFrame"
    assert X_unemployed.shape[1] == 1, "X_unemployed должен содержать только один столбец"
    
    # Выводим основную информацию о данных
    print(f"Подготовлены данные для моделей: размер X_all_groups: {X_all_groups.shape}, "
          f"X_unemployed: {X_unemployed.shape}, X_combined: {X_combined.shape}, y: {y.shape}")
    
    return X_all_groups, X_unemployed, X_combined, y