"""
Модуль для создания таблицы коэффициентов регрессии в формате Excel.

Содержит класс для генерации и отображения таблицы коэффициентов
регрессионной модели с форматированием, идентичным Excel, включая
доверительные интервалы и цветовое выделение значимых коэффициентов.
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

class CoefficientTable:
    """
    Класс для генерации и отображения таблицы коэффициентов регрессии.
    
    Создает таблицу коэффициентов в формате, идентичном Excel, с возможностью
    настройки стиля, фильтрации по значимости и экспорта данных.
    """
    def __init__(self, parent: tk.Widget, model_stats: Dict[str, Any], 
                 padx: int = 5, pady: int = 5, height: int = 10,
                 show_filter: bool = True, precision: int = 8):
        """
        Инициализация таблицы коэффициентов регрессии.
        
        Parameters:
        parent (tk.Widget): Родительский виджет, в котором будет размещена таблица
        model_stats (Dict[str, Any]): Статистические показатели модели
        padx (int): Отступ по горизонтали
        pady (int): Отступ по вертикали
        height (int): Высота таблицы в строках
        show_filter (bool): Показывать ли фильтр для отображения только значимых коэффициентов
        precision (int): Количество знаков после запятой для форматирования чисел
        """
        self.parent = parent
        self.model_stats = model_stats
        self.padx = padx
        self.pady = pady
        self.height = height
        self.show_filter = show_filter
        self.precision = precision
        self.show_significant_only = False
        
        # Создаем основной фрейм
        self.frame = tk.LabelFrame(parent, text="Коэффициенты", padx=10, pady=10)
        
        # Создаем фрейм для таблицы с прокруткой
        self.table_frame = tk.Frame(self.frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
        
        # Создаем и настраиваем таблицу коэффициентов
        self.create_coefficient_table()
        
        # Добавляем фильтр для отображения только значимых коэффициентов
        if self.show_filter:
            self.add_filter_controls()
    
    def create_coefficient_table(self):
        """Создает таблицу коэффициентов в формате Excel."""
        # Создаем Treeview для таблицы коэффициентов
        columns = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
        self.coef_table = ttk.Treeview(
            self.table_frame, 
            columns=columns, 
            show="headings", 
            height=self.height
        )
        
        # Настраиваем заголовки и ширину столбцов в формате Excel
        self.coef_table.heading("1", text="")
        self.coef_table.heading("2", text="Коэффициенты")
        self.coef_table.heading("3", text="Стандартная ошибка")
        self.coef_table.heading("4", text="t-статистика")
        self.coef_table.heading("5", text="P-Значение")
        self.coef_table.heading("6", text="Нижние 95%")
        self.coef_table.heading("7", text="Верхние 95%")
        self.coef_table.heading("8", text="Нижние 95.0%")
        self.coef_table.heading("9", text="Верхние 95.0%")
        
        self.coef_table.column("1", width=150, anchor=tk.W)
        self.coef_table.column("2", width=100, anchor=tk.CENTER)
        self.coef_table.column("3", width=150, anchor=tk.CENTER)
        self.coef_table.column("4", width=100, anchor=tk.CENTER)
        self.coef_table.column("5", width=100, anchor=tk.CENTER)
        self.coef_table.column("6", width=100, anchor=tk.CENTER)
        self.coef_table.column("7", width=100, anchor=tk.CENTER)
        self.coef_table.column("8", width=100, anchor=tk.CENTER)
        self.coef_table.column("9", width=100, anchor=tk.CENTER)
        
        # Настраиваем цвета для тегов
        self.coef_table.tag_configure("significant", background="#e6ffe6")  # Светло-зеленый
        self.coef_table.tag_configure("not_significant", background="#ffe6e6")  # Светло-красный
        
        # Заполняем таблицу данными
        self.populate_table()
        
        # Создаем полосы прокрутки
        scrollbar_x = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.coef_table.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        scrollbar_y = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.coef_table.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.coef_table.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
        self.coef_table.pack(fill=tk.BOTH, expand=True)
    
    def populate_table(self):
        """Заполняет таблицу коэффициентов данными из model_stats."""
        # Очищаем таблицу
        for item in self.coef_table.get_children():
            self.coef_table.delete(item)
        
        # Получаем данные для таблицы
        if 'excel_feature_names' in self.model_stats:
            feature_names = self.model_stats['excel_feature_names']
        else:
            feature_names = self.model_stats['feature_names']
            
        coefs = self.model_stats['coefficients']
        se_coefs = self.model_stats['se_coefficients']
        t_values = self.model_stats['t_values']
        p_values = self.model_stats['p_values']
        lower_ci = self.model_stats['lower_ci']
        upper_ci = self.model_stats['upper_ci']
        
        # Получаем 95.0% интервалы (идентичны 95%, но отображаются отдельно как в Excel)
        if 'lower_ci_95_0' in self.model_stats and 'upper_ci_95_0' in self.model_stats:
            lower_ci_95_0 = self.model_stats['lower_ci_95_0']
            upper_ci_95_0 = self.model_stats['upper_ci_95_0']
        else:
            lower_ci_95_0 = lower_ci
            upper_ci_95_0 = upper_ci
        
        # Заполняем таблицу данными
        for i, name in enumerate(feature_names):
            if i < len(coefs):
                # Определяем, значим ли коэффициент
                is_significant = p_values[i] < 0.05
                tag = "significant" if is_significant else "not_significant"
                
                # Если включен фильтр и коэффициент не значим, пропускаем
                if self.show_significant_only and not is_significant:
                    continue
                
                # Формируем строку с данными, форматируя числа
                coef_table_values = (
                    name, 
                    self.format_number(coefs[i]), 
                    self.format_number(se_coefs[i]), 
                    self.format_number(t_values[i]), 
                    self.format_p_value(p_values[i]),
                    self.format_number(lower_ci[i]),
                    self.format_number(upper_ci[i]),
                    self.format_number(lower_ci_95_0[i]),
                    self.format_number(upper_ci_95_0[i])
                )
                
                # Добавляем строку в таблицу
                self.coef_table.insert("", "end", values=coef_table_values, tags=(tag,))
    
    def add_filter_controls(self):
        """Добавляет элементы управления для фильтрации коэффициентов."""
        # Создаем фрейм для элементов управления
        filter_frame = tk.Frame(self.frame)
        filter_frame.pack(fill=tk.X, padx=self.padx, pady=self.pady)
        
        # Создаем переключатель для отображения только значимых коэффициентов
        self.show_significant_var = tk.BooleanVar(value=self.show_significant_only)
        filter_button = ttk.Checkbutton(
            filter_frame, 
            text="Показать только значимые коэффициенты (p < 0.05)", 
            variable=self.show_significant_var,
            command=self.toggle_significant_only
        )
        filter_button.pack(side=tk.LEFT, padx=5)
    
    def toggle_significant_only(self):
        """Переключает режим отображения только значимых коэффициентов."""
        self.show_significant_only = self.show_significant_var.get()
        self.populate_table()
    
    def format_number(self, value: Union[int, float]) -> str:
        """
        Форматирует числовое значение для отображения в таблице.
        
        Parameters:
        value (int, float): Числовое значение для форматирования
        
        Returns:
        str: Отформатированное значение
        """
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            # Форматирование чисел с фиксированной точностью (как в Excel)
            return f"{value:.{self.precision}f}"
        else:
            return str(value)
    
    def format_p_value(self, p_value: float) -> str:
        """
        Форматирует p-значение с добавлением звездочек для обозначения значимости.
        
        Parameters:
        p_value (float): p-значение для форматирования
        
        Returns:
        str: Отформатированное p-значение
        """
        if p_value < 0.001:
            return f"{p_value:.{self.precision}f} ***"
        elif p_value < 0.01:
            return f"{p_value:.{self.precision}f} **"
        elif p_value < 0.05:
            return f"{p_value:.{self.precision}f} *"
        else:
            return f"{p_value:.{self.precision}f}"
    
    def update_data(self, model_stats: Dict[str, Any]):
        """
        Обновляет данные в таблице коэффициентов.
        
        Parameters:
        model_stats (Dict[str, Any]): Новые статистические показатели модели
        """
        self.model_stats = model_stats
        self.populate_table()
    
    def get_selected_coefficient(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает данные выбранного коэффициента.
        
        Returns:
        Dict[str, Any] or None: Словарь с данными выбранного коэффициента или None, если ничего не выбрано
        """
        selected_items = self.coef_table.selection()
        if not selected_items:
            return None
        
        # Получаем значения выбранной строки
        values = self.coef_table.item(selected_items[0], 'values')
        
        # Индекс выбранного коэффициента
        feature_names = self.model_stats['excel_feature_names'] if 'excel_feature_names' in self.model_stats else self.model_stats['feature_names']
        try:
            index = feature_names.index(values[0])
        except ValueError:
            return None
        
        # Формируем словарь с данными коэффициента
        coef_data = {
            'name': values[0],
            'coefficient': self.model_stats['coefficients'][index],
            'se_coefficient': self.model_stats['se_coefficients'][index],
            't_value': self.model_stats['t_values'][index],
            'p_value': self.model_stats['p_values'][index],
            'lower_ci': self.model_stats['lower_ci'][index],
            'upper_ci': self.model_stats['upper_ci'][index],
            'index': index,
            'is_significant': self.model_stats['p_values'][index] < 0.05
        }
        
        return coef_data
    
    def set_selection_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Устанавливает функцию обратного вызова при выборе коэффициента.
        
        Parameters:
        callback (callable): Функция, которая будет вызвана при выборе коэффициента
        """
        def on_item_selected(event):
            coef_data = self.get_selected_coefficient()
            if coef_data:
                callback(coef_data)
        
        self.coef_table.bind('<<TreeviewSelect>>', on_item_selected)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Экспортирует таблицу коэффициентов в DataFrame.
        
        Returns:
        pandas.DataFrame: DataFrame с данными коэффициентов
        """
        # Получаем имена признаков
        if 'excel_feature_names' in self.model_stats:
            feature_names = self.model_stats['excel_feature_names']
        else:
            feature_names = self.model_stats['feature_names']
        
        # Создаем DataFrame с данными коэффициентов
        coef_data = {
            'Признак': feature_names,
            'Коэффициент': self.model_stats['coefficients'],
            'Стандартная_ошибка': self.model_stats['se_coefficients'],
            't-статистика': self.model_stats['t_values'],
            'P-значение': self.model_stats['p_values'],
            'Нижние_95%': self.model_stats['lower_ci'],
            'Верхние_95%': self.model_stats['upper_ci'],
            'Нижние_95.0%': self.model_stats['lower_ci_95_0'] if 'lower_ci_95_0' in self.model_stats else self.model_stats['lower_ci'],
            'Верхние_95.0%': self.model_stats['upper_ci_95_0'] if 'upper_ci_95_0' in self.model_stats else self.model_stats['upper_ci'],
            'Значимость': ['Значим' if p < 0.05 else 'Не значим' for p in self.model_stats['p_values']]
        }
        
        return pd.DataFrame(coef_data)
    
    def pack(self, **kwargs):
        """
        Упаковывает фрейм с таблицей.
        
        Parameters:
        **kwargs: Параметры для метода pack()
        """
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """
        Размещает фрейм с таблицей с помощью метода grid.
        
        Parameters:
        **kwargs: Параметры для метода grid()
        """
        self.frame.grid(**kwargs)
    
    def place(self, **kwargs):
        """
        Размещает фрейм с таблицей с помощью метода place.
        
        Parameters:
        **kwargs: Параметры для метода place()
        """
        self.frame.place(**kwargs)
    
    def get_frame(self) -> tk.LabelFrame:
        """
        Возвращает фрейм с таблицей.
        
        Returns:
        tk.LabelFrame: Фрейм, содержащий таблицу
        """
        return self.frame
    
    def get_significant_coefficients(self) -> List[Dict[str, Any]]:
        """
        Возвращает список значимых коэффициентов.
        
        Returns:
        List[Dict[str, Any]]: Список словарей с данными значимых коэффициентов
        """
        significant_coefs = []
        
        # Получаем имена признаков
        if 'excel_feature_names' in self.model_stats:
            feature_names = self.model_stats['excel_feature_names']
        else:
            feature_names = self.model_stats['feature_names']
        
        # Собираем данные значимых коэффициентов
        for i, p_value in enumerate(self.model_stats['p_values']):
            if p_value < 0.05:
                coef_data = {
                    'name': feature_names[i],
                    'coefficient': self.model_stats['coefficients'][i],
                    'se_coefficient': self.model_stats['se_coefficients'][i],
                    't_value': self.model_stats['t_values'][i],
                    'p_value': p_value,
                    'lower_ci': self.model_stats['lower_ci'][i],
                    'upper_ci': self.model_stats['upper_ci'][i],
                    'index': i
                }
                significant_coefs.append(coef_data)
        
        return significant_coefs

# Пример использования:
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Пример таблицы коэффициентов")
    
    # Пример данных
    model_stats = {
        'feature_names': ['Y-пересечение', 'X1', 'X2', 'X3'],
        'excel_feature_names': ['Y-пересечение', 'Переменная X 1', 'Переменная X 2', 'Переменная X 3'],
        'coefficients': [123.456, 7.890, -2.345, 0.678],
        'se_coefficients': [12.345, 1.234, 0.567, 0.123],
        't_values': [10.0, 6.4, -4.13, 5.5],
        'p_values': [0.0001, 0.0023, 0.0456, 0.078],
        'lower_ci': [100.0, 5.5, -4.5, 0.4],
        'upper_ci': [150.0, 10.2, -0.5, 0.9]
    }
    
    # Создаем таблицу
    coef_table = CoefficientTable(root, model_stats)
    coef_table.set_selection_callback(lambda data: print(f"Выбран коэффициент: {data['name']}"))
    coef_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    root.mainloop()