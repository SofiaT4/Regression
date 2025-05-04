"""
Модуль для отображения графиков зависимости ВВП от экономических показателей.

Содержит класс DependencyViewer, который создает интерфейс для выбора
и отображения различных типов зависимостей между ВВП и другими факторами.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Dict, List, Union, Optional, Any

from ui.components.ui_helpers import center_window
from utils.visualization.dependency_plots import (
    create_scatter_plot,
    create_multi_scatter_plot,
    create_3d_plot,
    create_heatmap_plot,
    create_partial_dependence_plot
)


class DependencyViewer(ttk.Frame):
    """
    Класс для отображения графиков зависимости ВВП от других экономических показателей.
    
    Позволяет пользователю создавать графики:
    - Диаграммы рассеяния (ВВП vs одна переменная)
    - Множественные диаграммы рассеяния
    - 3D-графики зависимостей
    - Тепловые карты корреляций
    - Графики частичной зависимости
    """
    def __init__(self, parent, df, stats_dict, model_dict, year_range, age_groups=None, y=None):
        """
        Инициализирует виджет просмотра зависимостей ВВП от других показателей.
        
        Args:
            parent: Родительский виджет
            df: DataFrame с данными
            stats_dict: Словарь со статистикой моделей
            model_dict: Словарь с моделями
            year_range: Кортеж с диапазоном лет (min_year, max_year)
            age_groups: Список возрастных групп (опционально)
            y: Целевая переменная (ВВП) (опционально)
        """
        super().__init__(parent)
        self.parent = parent
        self.df = df
        self.stats_dict = stats_dict
        self.model_dict = model_dict
        self.min_year, self.max_year = year_range
        self.age_groups = age_groups if age_groups else []
        self.y = y
        
        # Сохраняем данные о годах
        self.years = list(range(self.min_year, self.max_year + 1))
        
        # Получаем имена переменных из данных
        self.feature_names = self._get_feature_names()
        
        # Добавляем переменную ВВП для отображения в списках
        self.graph_variables = ['ВВП'] + self.feature_names
        
        # Атрибуты для хранения виджетов
        self.canvas = None
        self.toolbar = None
        self.figure = None
        self.plot_type = tk.StringVar(value="scatter")
        self.current_year = tk.StringVar(value=str(self.max_year))
        self.x_var = tk.StringVar()  # Переменная для оси X
        self.y_var = tk.StringVar(value="ВВП")  # Переменная для оси Y
        self.z_var = tk.StringVar()  # Переменная для оси Z (для 3D)
        self.selected_features = []  # Выбранные признаки для множественной регрессии
        
        # Создаем интерфейс
        self._create_widgets()
        
        # Устанавливаем начальные значения
        if self.feature_names:
            self.x_var.set(self.feature_names[0])
            if len(self.feature_names) > 1:
                self.z_var.set(self.feature_names[1])
    
    def _get_feature_names(self):
        """
        Получает список имен признаков из данных, исключая ВВП.
        
        Returns:
            list: Список имен признаков
        """
        # Предполагаем, что данные содержат колонки с годами и экономическими переменными
        # Исключаем колонки с годами и ВВП
        cols = [col for col in self.df.columns if col != 'ВВП' and not str(col).isdigit()]
        return cols
    
    def _create_widgets(self):
        """Создает интерфейс просмотра зависимостей."""
        # Основной контейнер с сеткой
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель настроек
        self.control_frame = ttk.LabelFrame(main_frame, text="Настройки")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Frame для выбора типа графика
        plot_type_frame = ttk.Frame(self.control_frame)
        plot_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(plot_type_frame, text="Тип графика:").pack(anchor='w')
        
        # Радиокнопки для выбора типа графика
        types = [
            ("Диаграмма рассеяния", "scatter"),
            ("Множественная диаграмма", "multi_scatter"),
            ("3D график", "3d"),
            ("Тепловая карта", "heatmap"),
            ("Частичная зависимость", "partial_dependence")
        ]
        
        for text, value in types:
            ttk.Radiobutton(
                plot_type_frame, 
                text=text, 
                value=value, 
                variable=self.plot_type,
                command=self.on_plot_type_change
            ).pack(anchor='w', padx=10)
        
        # Создаем контейнер для переменных осей, который будет меняться в зависимости от типа графика
        self.variables_frame = ttk.LabelFrame(self.control_frame, text="Переменные")
        self.variables_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Кнопка для создания графика
        ttk.Button(
            self.control_frame, 
            text="Построить график", 
            command=self.create_plot
        ).pack(fill=tk.X, padx=5, pady=10)
        
        # Фрейм для графика справа
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Инициализируем пустой график
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем тулбар для графика
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        
        # Инициализируем виджеты для выбора переменных
        self.setup_variable_controls()
    
    def setup_variable_controls(self):
        """
        Настраивает элементы управления для выбора переменных
        в зависимости от выбранного типа графика.
        """
        # Очищаем предыдущие виджеты
        for widget in self.variables_frame.winfo_children():
            widget.destroy()
        
        plot_type = self.plot_type.get()
        
        # Создаем виджеты для каждого типа графика
        if plot_type == "scatter":
            # Для диаграммы рассеяния нужны X и Y
            ttk.Label(self.variables_frame, text="Ось X:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            x_cb = ttk.Combobox(self.variables_frame, textvariable=self.x_var, values=self.feature_names, state="readonly")
            x_cb.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            
            # Выбор года для анализа
            ttk.Label(self.variables_frame, text="Год:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            years_cb = ttk.Combobox(self.variables_frame, textvariable=self.current_year, values=[str(y) for y in self.years], state="readonly")
            years_cb.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
            
        elif plot_type == "multi_scatter":
            # Для множественной диаграммы нужны несколько X и один Y
            ttk.Label(self.variables_frame, text="Выберите переменные для осей X:").grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
            
            self.feature_vars = []
            for i, feature in enumerate(self.feature_names):
                var = tk.BooleanVar(value=False)
                self.feature_vars.append(var)
                ttk.Checkbutton(self.variables_frame, text=feature, variable=var).grid(row=i+1, column=0, sticky='w', padx=15, pady=2)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=len(self.feature_names)+1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=len(self.feature_names)+1, column=1, sticky='ew', padx=5, pady=2)
            
        elif plot_type == "3d":
            # Для 3D графика нужны X, Y, Z
            ttk.Label(self.variables_frame, text="Ось X:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            x_cb = ttk.Combobox(self.variables_frame, textvariable=self.x_var, values=self.feature_names, state="readonly")
            x_cb.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            
            ttk.Label(self.variables_frame, text="Ось Z:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            z_cb = ttk.Combobox(self.variables_frame, textvariable=self.z_var, values=self.feature_names, state="readonly")
            z_cb.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
            
        elif plot_type == "heatmap":
            # Для тепловой карты не нужны дополнительные параметры
            ttk.Label(self.variables_frame, text="Тепловая карта показывает корреляции\nмежду всеми экономическими показателями").grid(row=0, column=0, padx=5, pady=10)
            
        elif plot_type == "partial_dependence":
            # Для частичной зависимости нужна переменная X
            ttk.Label(self.variables_frame, text="Выберите переменную:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            x_cb = ttk.Combobox(self.variables_frame, textvariable=self.x_var, values=self.feature_names, state="readonly")
            x_cb.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
            
            ttk.Label(self.variables_frame, text="Модель:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            model_cb = ttk.Combobox(self.variables_frame, values=list(self.model_dict.keys()), state="readonly")
            model_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            model_cb.current(0)
            self.model_var = model_cb
    
    def on_plot_type_change(self):
        """Обработчик изменения типа графика."""
        self.setup_variable_controls()
    
    def get_selected_features(self):
        """
        Получает список выбранных признаков для множественной регрессии.
        
        Returns:
            list: Список выбранных имен признаков
        """
        selected = []
        if hasattr(self, 'feature_vars'):
            for i, var in enumerate(self.feature_vars):
                if var.get():
                    selected.append(self.feature_names[i])
        return selected
    
    def create_plot(self):
        """
        Создает и отображает выбранный тип графика.
        """
        plot_type = self.plot_type.get()
        
        # Очищаем текущий график
        self.figure.clear()
        
        # Общие данные
        # ВВП всегда на оси Y для обычных графиков
        y_data = self.y  # Используем переданную переменную y вместо поиска в df
        
        try:
            if plot_type == "scatter":
                # Диаграмма рассеяния
                x_feature = self.x_var.get()
                year = int(self.current_year.get())
                
                x_data = self.df[x_feature]
                
                fig = create_scatter_plot(
                    x_data=x_data,
                    y_data=y_data,
                    feature_name=x_feature,
                    year_data=self.years,
                    highlight_year=year
                )
                self.figure = fig
                
            elif plot_type == "multi_scatter":
                # Множественная диаграмма рассеяния
                selected_features = self.get_selected_features()
                
                if not selected_features:
                    messagebox.showwarning("Предупреждение", "Выберите хотя бы один признак для анализа")
                    return
                
                X_selected = self.df[selected_features]
                
                fig = create_multi_scatter_plot(
                    X=X_selected,
                    y=y_data,
                    feature_names=selected_features,
                    target_name="ВВП"
                )
                self.figure = fig
                
            elif plot_type == "3d":
                # 3D график
                x_feature = self.x_var.get()
                z_feature = self.z_var.get()
                
                if x_feature == z_feature:
                    messagebox.showwarning("Предупреждение", "Выберите разные признаки для осей X и Z")
                    return
                
                X_selected = self.df[[x_feature, z_feature]]
                
                fig = create_3d_plot(
                    X=X_selected,
                    y=y_data,
                    feature_names=[x_feature, z_feature],
                    target_name="ВВП",
                    years=self.years
                )
                self.figure = fig
                
            elif plot_type == "heatmap":
                # Тепловая карта корреляций
                # Создаем DataFrame с признаками и целевой переменной
                data_for_corr = self.df[self.feature_names].copy()
                data_for_corr["ВВП"] = y_data
                
                fig = create_heatmap_plot(
                    data=data_for_corr,
                    title="Корреляции между экономическими показателями"
                )
                self.figure = fig
                
            elif plot_type == "partial_dependence":
                # График частичной зависимости
                x_feature = self.x_var.get()
                model_name = self.model_var.get()
                
                if not model_name:
                    messagebox.showwarning("Предупреждение", "Выберите модель для анализа")
                    return
                
                # В зависимости от структуры модели, мы можем получить доступ к модели по-разному
                model = self.model_dict[model_name]
                
                # Проверяем, есть ли метод get_model()
                if hasattr(model, 'get_model'):
                    sklearn_model = model.get_model()
                    feature_names = model.get_feature_names() if hasattr(model, 'get_feature_names') else None
                    X_model = model.get_x() if hasattr(model, 'get_x') else None
                else:
                    # Если это просто объект модели sklearn
                    sklearn_model = model
                    # Пытаемся найти имена признаков и данные из других источников
                    if model_name == 'all_groups':
                        feature_names = self.age_groups if hasattr(self, 'age_groups') else None
                        X_model = self.df[feature_names] if feature_names else None
                    elif model_name == 'unemployed':
                        feature_names = ['Численность безработных']
                        X_model = self.df[feature_names] if 'Численность безработных' in self.df.columns else None
                    elif model_name == 'combined':
                        if hasattr(self, 'age_groups'):
                            feature_names = self.age_groups + ['Численность безработных']
                            X_model = self.df[feature_names] if all(f in self.df.columns for f in feature_names) else None
                        else:
                            feature_names = None
                            X_model = None
                
                # Если мы не смогли найти имена признаков или данные
                if not feature_names or not X_model:
                    messagebox.showwarning("Предупреждение", 
                                         f"Не удалось получить данные для модели {model_name}")
                    return
                
                # Находим индекс признака
                feature_idx = None
                if x_feature in feature_names:
                    feature_idx = feature_names.index(x_feature)
                else:
                    messagebox.showwarning("Предупреждение", 
                                         f"Признак {x_feature} не используется в модели {model_name}")
                    return
                
                fig = create_partial_dependence_plot(
                    model=sklearn_model,
                    X=X_model,
                    feature_idx=feature_idx,
                    feature_name=x_feature,
                    target_name="ВВП"
                )
                self.figure = fig
            
            # Обновляем холст
            self.canvas.figure = self.figure
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при создании графика: {str(e)}") 