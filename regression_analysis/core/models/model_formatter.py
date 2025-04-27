"""
Модуль для форматирования и представления регрессионных моделей.

Содержит функции для форматирования уравнений регрессии, 
преобразования статистических данных в читаемый вид и упрощения 
представления признаков для отображения.
"""

import re
from typing import Dict, List, Any, Optional, Union, Tuple

def format_equation_for_display(
    model_stats: Dict[str, Any], 
    current_model: str, 
    age_group_label: str, 
    additional_labels: List[str]
) -> str:
    """
    Форматирует уравнение регрессии для отображения в интерфейсе в формате Excel с улучшенным форматированием.
    
    Parameters:
    model_stats (dict): Статистические показатели модели
    current_model (str): Тип текущей модели ('all_groups', 'unemployed', 'combined')
    age_group_label (str): Метка для возрастных групп
    additional_labels (list): Дополнительные метки (например, "Безработица")
    
    Returns:
    str: Отформатированное уравнение регрессии
    """
    coefs = model_stats['coefficients']
    
    # Используем Excel-форматированные имена, если доступны
    if 'excel_feature_names' in model_stats:
        feature_names = model_stats['excel_feature_names']
    else:
        feature_names = model_stats['feature_names']
    
    # Функция для форматирования чисел в зависимости от их величины
    def format_number(value):
        if abs(value) >= 1000000:
            return f"{value:.2e}"  # Научная нотация для больших чисел
        elif abs(value) < 0.001:
            return f"{value:.6f}"  # Больше десятичных знаков для очень маленьких чисел
        else:
            return f"{value:.4f}"  # Стандартное форматирование
    
    # Форматируем значение константы
    intercept = coefs[0]
    intercept_str = format_number(intercept)
    
    if current_model == 'unemployed':
        # Для модели с одной переменной (безработица)
        if len(coefs) > 1:
            slope = coefs[1]
            slope_str = format_number(slope)
            
            # Используем имя переменной или "Безработица"
            variable_name = additional_labels[0] if additional_labels else "Безработица"
            
            # Форматируем знак перед константой (как в Excel)
            sign = "+" if intercept >= 0 else "-"
            abs_intercept = abs(intercept)
            abs_intercept_str = format_number(abs_intercept)
            
            # Уравнение в формате Excel: y = mx + b
            equation = f"y = {slope_str}·{variable_name} {sign} {abs_intercept_str}"
        else:
            equation = f"y = {intercept_str}"  # Только константа, если нет коэффициентов
            
    elif current_model == 'all_groups':
        # Для модели с множественными переменными (все возрастные группы)
        # Начинаем уравнение с константы
        equation = f"y = {intercept_str}"
        
        # Добавляем каждый коэффициент, игнорируя константу (индекс 0)
        for i in range(1, len(coefs)):
            coef = coefs[i]
            coef_str = format_number(abs(coef))
            
            # Используем реальные имена признаков, если они доступны
            if i < len(feature_names):
                # Упрощаем имя признака, если возможно
                variable_name = feature_names[i]
                if len(feature_names) > 10:  # Если много признаков, используем краткие имена
                    variable_name = f"X{i}"
            else:
                variable_name = f"X{i}"
            
            # Форматируем знак коэффициента
            if coef >= 0:
                equation += f" + {coef_str}·{variable_name}"
            else:
                equation += f" - {coef_str}·{variable_name}"
        
    else:  # combined model
        # Для комбинированной модели (начинаем с константы)
        equation = f"y = {intercept_str}"
        
        # Добавляем каждый коэффициент, игнорируя константу (индекс 0)
        for i in range(1, len(coefs)):
            coef = coefs[i]
            coef_str = format_number(abs(coef))
            
            # Используем реальные имена признаков, если они доступны
            if i < len(feature_names):
                variable_name = feature_names[i]
                
                # Если это последний коэффициент и есть дополнительные метки,
                # используем метку "Безработица" для последнего коэффициента в комбинированной модели
                if i == len(coefs) - 1 and additional_labels:
                    variable_name = additional_labels[0]
            else:
                variable_name = f"X{i}"
            
            # Форматируем знак коэффициента
            if coef >= 0:
                equation += f" + {coef_str}·{variable_name}"
            else:
                equation += f" - {coef_str}·{variable_name}"
    
    return equation

def format_equation_for_charts(model_stats: Dict[str, Any], current_model: str) -> str:
    """
    Форматирует уравнение регрессии для отображения на графиках с улучшенным форматированием.
    
    Parameters:
    model_stats (dict): Статистические показатели модели
    current_model (str): Тип текущей модели ('all_groups', 'unemployed', 'combined')
    
    Returns:
    str: Упрощенное уравнение для графиков
    """
    coefs = model_stats['coefficients']
    
    # Функция для форматирования чисел в зависимости от их величины
    def format_number(value):
        if abs(value) >= 1000000:
            return f"{value:.2e}"  # Научная нотация для больших чисел
        elif abs(value) < 0.001:
            return f"{value:.6f}"  # Больше десятичных знаков для очень маленьких чисел
        else:
            return f"{value:.4f}"  # Стандартное форматирование
    
    if current_model == 'unemployed':
        # Для модели с одной переменной (безработица)
        intercept = coefs[0]
        
        if len(coefs) > 1:
            slope = coefs[1]
            slope_str = format_number(slope)
            intercept_str = format_number(intercept)
            equation = f"y = {slope_str}x + {intercept_str}"
        else:
            intercept_str = format_number(intercept)
            equation = f"y = {intercept_str}"
            
    elif current_model == 'all_groups':
        # Для графиков модели с возрастными группами
        intercept = coefs[0]
        intercept_str = format_number(intercept)
        
        # Если есть коэффициенты (кроме константы), находим наиболее значимые
        if len(coefs) > 1:
            # Находим два наибольших по модулю коэффициента для упрощенной модели
            if len(coefs) > 3:  # Если коэффициентов много
                sorted_indices = sorted(range(1, len(coefs)), key=lambda i: abs(coefs[i]), reverse=True)
                
                # Выбираем два наиболее значимых коэффициента
                first_idx = sorted_indices[0]
                second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
                
                first_coef = format_number(coefs[first_idx])
                equation = f"y = {first_coef}x₁"
                
                if second_idx is not None:
                    second_coef = coefs[second_idx]
                    second_coef_str = format_number(abs(second_coef))
                    
                    if second_coef >= 0:
                        equation += f" + {second_coef_str}x₂"
                    else:
                        equation += f" - {second_coef_str}x₂"
                
                if intercept >= 0:
                    equation += f" + {intercept_str}"
                else:
                    intercept_abs_str = format_number(abs(intercept))
                    equation += f" - {intercept_abs_str}"
            else:
                # Если коэффициентов мало, включаем их все
                equation = f"y = {intercept_str}"
                
                for i in range(1, len(coefs)):
                    coef = coefs[i]
                    coef_str = format_number(abs(coef))
                    
                    if coef >= 0:
                        equation += f" + {coef_str}x{i}"
                    else:
                        equation += f" - {coef_str}x{i}"
        else:
            # Если коэффициентов нет, только константа
            equation = f"y = {intercept_str}"
            
    else:  # combined model
        intercept = coefs[0]
        intercept_str = format_number(intercept)
        
        # Для комбинированной модели показываем вклад возрастных групп и безработицы
        if len(coefs) > 2:  # Если есть коэффициенты и возрастных групп, и безработицы
            # Получаем коэффициент безработицы (обычно последний в списке)
            unemployment_idx = len(coefs) - 1
            unemployment_coef = coefs[unemployment_idx]
            unemployment_coef_str = format_number(abs(unemployment_coef))
            
            # Находим наиболее значимую возрастную группу
            age_group_indices = list(range(1, unemployment_idx))
            if age_group_indices:
                age_idx = max(age_group_indices, key=lambda i: abs(coefs[i]))
                age_coef = coefs[age_idx]
                age_coef_str = format_number(abs(age_coef))
                
                # Формируем уравнение с двумя основными предикторами
                equation = f"y = "
                
                # Добавляем коэффициент возрастной группы
                if age_coef >= 0:
                    equation += f"{age_coef_str}x₁"
                else:
                    equation += f"-{age_coef_str}x₁"
                
                # Добавляем коэффициент безработицы
                if unemployment_coef >= 0:
                    equation += f" + {unemployment_coef_str}x₂"
                else:
                    equation += f" - {unemployment_coef_str}x₂"
                
                # Добавляем константу
                if intercept >= 0:
                    equation += f" + {intercept_str}"
                else:
                    intercept_abs_str = format_number(abs(intercept))
                    equation += f" - {intercept_abs_str}"
            else:
                # Если нет возрастных групп, только безработица
                equation = f"y = "
                
                if unemployment_coef >= 0:
                    equation += f"{unemployment_coef_str}x"
                else:
                    equation += f"-{unemployment_coef_str}x"
                
                if intercept >= 0:
                    equation += f" + {intercept_str}"
                else:
                    intercept_abs_str = format_number(abs(intercept))
                    equation += f" - {intercept_abs_str}"
        else:
            # Если мало коэффициентов, простая модель
            if len(coefs) > 1:
                coef = coefs[1]
                coef_str = format_number(abs(coef))
                
                equation = f"y = "
                if coef >= 0:
                    equation += f"{coef_str}x"
                else:
                    equation += f"-{coef_str}x"
                
                if intercept >= 0:
                    equation += f" + {intercept_str}"
                else:
                    intercept_abs_str = format_number(abs(intercept))
                    equation += f" - {intercept_abs_str}"
            else:
                equation = f"y = {intercept_str}"
    
    return equation

def simplify_feature_name(name: str) -> str:
    """
    Упрощает имя признака, удаляя ненужные префиксы и разметку.
    
    Parameters:
    name (str): Исходное имя признака
    
    Returns:
    str: Упрощенное имя признака
    """
    # Удаляем "Unnamed:" и подобные префиксы
    name = re.sub(r'Unnamed:\s*\d+_level_\d+\s*-\s*', '', name)
    
    # Пробуем выделить только диапазон возраста (например, 25-34)
    age_match = re.search(r'(\d+[-–]\d+)', name)
    if age_match:
        return age_match.group(1)
    
    # Сокращаем длинные имена
    if len(name) > 40:
        words = name.split()
        if len(words) > 3:
            # Оставляем только первые и последние слова для длинных названий
            name = ' '.join(words[:2] + ['...'] + words[-1:])
    
    # Возвращаем исходное имя, если не удалось упростить
    return name

def format_coefficient_table(
    coefficients: List[float], 
    feature_names: List[str], 
    p_values: Optional[List[float]] = None, 
    se_coefficients: Optional[List[float]] = None,
    t_values: Optional[List[float]] = None,
    lower_ci: Optional[List[float]] = None,
    upper_ci: Optional[List[float]] = None,
    significant_only: bool = False,
    alpha: float = 0.05
) -> Dict[str, List[Union[str, float]]]:
    """
    Форматирует коэффициенты регрессии для отображения в таблице.
    
    Parameters:
    coefficients (List[float]): Коэффициенты регрессии
    feature_names (List[str]): Названия признаков
    p_values (List[float], optional): p-значения для каждого коэффициента
    se_coefficients (List[float], optional): Стандартные ошибки коэффициентов
    t_values (List[float], optional): t-статистики коэффициентов
    lower_ci (List[float], optional): Нижние границы доверительных интервалов
    upper_ci (List[float], optional): Верхние границы доверительных интервалов
    significant_only (bool): Показывать только значимые коэффициенты
    alpha (float): Уровень значимости для выделения значимых коэффициентов
    
    Returns:
    Dict[str, List[Union[str, float]]]: Отформатированные данные для таблицы
    """
    # Подготавливаем данные для таблицы
    table_data = {
        'Признак': [],
        'Коэффициент': [],
        'Стандартная ошибка': [],
        't-статистика': [],
        'P-значение': [],
        'Значимость': [],
    }
    
    # Добавляем колонки для доверительных интервалов, если они предоставлены
    if lower_ci is not None and upper_ci is not None:
        table_data['Нижний 95% CI'] = []
        table_data['Верхний 95% CI'] = []
        table_data['Нижний 95.0% CI'] = []
        table_data['Верхний 95.0% CI'] = []
    
    # Заполняем таблицу данными
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        # Проверяем значимость, если p-значения предоставлены
        is_significant = False
        if p_values is not None and i < len(p_values):
            is_significant = p_values[i] < alpha
        
        # Если нужно показывать только значимые коэффициенты, пропускаем незначимые
        if significant_only and not is_significant:
            continue
        
        # Добавляем данные в таблицу
        table_data['Признак'].append(name)
        table_data['Коэффициент'].append(f"{coef:.6f}")
        
        # Добавляем стандартную ошибку, если она предоставлена
        if se_coefficients is not None and i < len(se_coefficients):
            table_data['Стандартная ошибка'].append(f"{se_coefficients[i]:.6f}")
        else:
            table_data['Стандартная ошибка'].append("")
        
        # Добавляем t-статистику, если она предоставлена
        if t_values is not None and i < len(t_values):
            table_data['t-статистика'].append(f"{t_values[i]:.4f}")
        else:
            table_data['t-статистика'].append("")
        
        # Добавляем p-значение и значимость, если они предоставлены
        if p_values is not None and i < len(p_values):
            p_value = p_values[i]
            if p_value < 0.001:
                p_text = "< 0.001 ***"
            elif p_value < 0.01:
                p_text = f"{p_value:.4f} **"
            elif p_value < 0.05:
                p_text = f"{p_value:.4f} *"
            else:
                p_text = f"{p_value:.4f}"
            
            table_data['P-значение'].append(p_text)
            table_data['Значимость'].append("Да" if is_significant else "Нет")
        else:
            table_data['P-значение'].append("")
            table_data['Значимость'].append("")
        
        # Добавляем доверительные интервалы, если они предоставлены
        if lower_ci is not None and upper_ci is not None and i < len(lower_ci) and i < len(upper_ci):
            table_data['Нижний 95% CI'].append(f"{lower_ci[i]:.4f}")
            table_data['Верхний 95% CI'].append(f"{upper_ci[i]:.4f}")
            # Добавляем 95.0% интервалы (идентичны 95%, но отображаются отдельно как в Excel)
            table_data['Нижний 95.0% CI'].append(f"{lower_ci[i]:.4f}")
            table_data['Верхний 95.0% CI'].append(f"{upper_ci[i]:.4f}")
    
    return table_data

def get_model_summary(model_stats: Dict[str, Any], model_type: str) -> Dict[str, str]:
    """
    Создает сводку модели с основными показателями.
    
    Parameters:
    model_stats (Dict[str, Any]): Статистические показатели модели
    model_type (str): Тип модели ('all_groups', 'unemployed', 'combined')
    
    Returns:
    Dict[str, str]: Словарь с основными показателями модели
    """
    # Названия моделей для отображения
    model_names = {
        'all_groups': 'Модель от численности рабочих',
        'unemployed': 'Модель от безработицы',
        'combined': 'Комбинированная модель'
    }
    
    # Получаем основные показатели
    r2 = model_stats.get('r2', 0)
    adjusted_r2 = model_stats.get('adjusted_r2', 0)
    f_statistic = model_stats.get('f_statistic', 0)
    p_value_f = model_stats.get('p_value_f', 1)
    observations = model_stats.get('observations', 0)
    
    # Вычисляем количество значимых коэффициентов
    significant_coefs = 0
    if 'p_values' in model_stats and 'coefficients' in model_stats:
        for i, p in enumerate(model_stats['p_values']):
            if i > 0 and p < 0.05:  # Пропускаем константу (i=0)
                significant_coefs += 1
    
    # Оценка качества модели
    quality = ""
    if r2 > 0.9:
        quality = "Очень высокое"
    elif r2 > 0.7:
        quality = "Высокое"
    elif r2 > 0.5:
        quality = "Среднее"
    elif r2 > 0.3:
        quality = "Низкое"
    else:
        quality = "Очень низкое"
    
    # Оценка значимости модели
    significance = ""
    if p_value_f < 0.001:
        significance = "Очень высокая (p < 0.001)"
    elif p_value_f < 0.01:
        significance = "Высокая (p < 0.01)"
    elif p_value_f < 0.05:
        significance = "Средняя (p < 0.05)"
    else:
        significance = "Незначимая (p ≥ 0.05)"
    
    # Формируем сводку
    summary = {
        'Тип модели': model_names.get(model_type, model_type),
        'R-квадрат': f"{r2:.4f}",
        'Скорректированный R-квадрат': f"{adjusted_r2:.4f}",
        'F-статистика': f"{f_statistic:.4f}",
        'Значимость F': f"{p_value_f:.6f}",
        'Количество наблюдений': str(observations),
        'Количество значимых предикторов': str(significant_coefs),
        'Качество модели': quality,
        'Значимость модели': significance
    }
    
    return summary

def format_statistic_value(value: Union[float, int, str], 
                          format_type: str = "standard", 
                          precision: int = 4) -> str:
    """
    Форматирует статистическое значение для отображения.
    
    Parameters:
    value (float, int, str): Значение для форматирования
    format_type (str): Тип форматирования ('standard', 'percent', 'scientific')
    precision (int): Точность (количество знаков после запятой)
    
    Returns:
    str: Отформатированное значение
    """
    if value is None or value == "":
        return "—"  # Em dash для пустых значений
    
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    
    if format_type == "percent":
        # Процентный формат
        return f"{value * 100:.{precision}f}%"
    
    elif format_type == "scientific":
        # Научный формат
        if abs(value) >= 1e6 or abs(value) <= 1e-6 and value != 0:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    
    else:  # standard
        # Стандартный формат
        if abs(value) >= 1e6:
            # Для больших чисел используем миллионы
            return f"{value/1e6:.{precision}f}M"
        elif abs(value) >= 1e3:
            # Для тысяч
            return f"{value/1e3:.{precision}f}k"
        elif abs(value) < 1e-4 and value != 0:
            # Для очень маленьких чисел
            return f"{value:.{precision}e}"
        else:
            # Обычное форматирование
            return f"{value:.{precision}f}"