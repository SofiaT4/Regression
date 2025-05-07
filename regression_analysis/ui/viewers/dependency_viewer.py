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

from ui.components.ui_helpers import center_window, RussianNavigationToolbar
from utils.visualization.dependency_plots import (
    create_scatter_plot,
    create_multi_scatter_plot,
    create_3d_plot,
    create_heatmap_plot,
    create_partial_dependence_plot
)
from ui.components.theme_manager import DARK_THEME, get_chart_colors


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
        
        # Применяем темную тему для matplotlib
        self.apply_dark_theme_to_matplotlib()
        
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
    
    def apply_dark_theme_to_matplotlib(self):
        """Применяет темную тему к графикам matplotlib"""
        plt.style.use('dark_background')
        
        # Настраиваем цвета в соответствии с DARK_THEME
        plt.rcParams['figure.facecolor'] = DARK_THEME['primary']
        plt.rcParams['axes.facecolor'] = DARK_THEME['bg']
        plt.rcParams['axes.edgecolor'] = DARK_THEME['neutral']
        plt.rcParams['axes.labelcolor'] = DARK_THEME['neutral']
        plt.rcParams['axes.titlecolor'] = DARK_THEME['text_light']
        plt.rcParams['xtick.color'] = DARK_THEME['neutral']
        plt.rcParams['ytick.color'] = DARK_THEME['neutral']
        plt.rcParams['text.color'] = DARK_THEME['neutral']
        plt.rcParams['grid.color'] = DARK_THEME['bg_light']
        plt.rcParams['grid.alpha'] = 0.3
        
        # Настраиваем цвета для графиков
        plt.rcParams['lines.color'] = DARK_THEME['accent']
        plt.rcParams['patch.facecolor'] = DARK_THEME['accent']
        plt.rcParams['boxplot.boxprops.color'] = DARK_THEME['neutral']
        plt.rcParams['boxplot.capprops.color'] = DARK_THEME['neutral']
        plt.rcParams['boxplot.flierprops.color'] = DARK_THEME['neutral']
        plt.rcParams['boxplot.flierprops.markeredgecolor'] = DARK_THEME['neutral']
        plt.rcParams['boxplot.whiskerprops.color'] = DARK_THEME['neutral']
        
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
        
        # Создаем фрейм для графика справа
        self.graph_frame = tk.Frame(main_frame, bg=DARK_THEME['primary'])
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Инициализируем пустой график с горизонтальным соотношением сторон и темной темой
        self.figure = Figure(figsize=(10, 5.5), dpi=100, facecolor=DARK_THEME['primary'])
        
        # Создаем контейнер для графика
        self.canvas_container = tk.Frame(self.graph_frame, bg=DARK_THEME['primary'])
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Создаем холст для отображения графика
        self.canvas = FigureCanvasTkAgg(self.figure, self.canvas_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Создаем фрейм для тулбара и размещаем его внизу
        self.toolbar_frame = tk.Frame(self.graph_frame, bg=DARK_THEME['primary'])
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Создаем экземпляр тулбара с правильной привязкой к canvas
        self.toolbar = RussianNavigationToolbar(self.canvas, self.toolbar_frame, theme=DARK_THEME)
        
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
            self._style_combobox(x_cb)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(y_cb)
            
            # Выбор года для анализа
            ttk.Label(self.variables_frame, text="Год:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            years_cb = ttk.Combobox(self.variables_frame, textvariable=self.current_year, values=[str(y) for y in self.years], state="readonly")
            years_cb.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(years_cb)
            
        elif plot_type == "multi_scatter":
            # Для множественной диаграммы нужны несколько X и один Y
            ttk.Label(self.variables_frame, text="Выберите переменные для осей X:").grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
            
            self.feature_vars = []
            for i, feature in enumerate(self.feature_names):
                var = tk.BooleanVar(value=False)
                self.feature_vars.append(var)
                cb = ttk.Checkbutton(self.variables_frame, text=feature, variable=var)
                cb.grid(row=i+1, column=0, sticky='w', padx=15, pady=2)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=len(self.feature_names)+1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=len(self.feature_names)+1, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(y_cb)
            
        elif plot_type == "3d":
            # Для 3D графика нужны X, Y, Z
            ttk.Label(self.variables_frame, text="Ось X:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            x_cb = ttk.Combobox(self.variables_frame, textvariable=self.x_var, values=self.feature_names, state="readonly")
            x_cb.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(x_cb)
            
            ttk.Label(self.variables_frame, text="Ось Y:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            y_cb = ttk.Combobox(self.variables_frame, textvariable=self.y_var, values=[self.graph_variables[0]], state="readonly")
            y_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(y_cb)
            
            ttk.Label(self.variables_frame, text="Ось Z:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
            z_cb = ttk.Combobox(self.variables_frame, textvariable=self.z_var, values=self.feature_names, state="readonly")
            z_cb.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(z_cb)
            
        elif plot_type == "heatmap":
            # Для тепловой карты не нужны дополнительные параметры
            ttk.Label(self.variables_frame, text="Тепловая карта показывает корреляции\nмежду всеми экономическими показателями").grid(row=0, column=0, padx=5, pady=10)
            
        elif plot_type == "partial_dependence":
            # Для частичной зависимости нужна переменная X
            ttk.Label(self.variables_frame, text="Выберите переменную:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
            x_cb = ttk.Combobox(self.variables_frame, textvariable=self.x_var, values=self.feature_names, state="readonly")
            x_cb.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
            self._style_combobox(x_cb)
            
            ttk.Label(self.variables_frame, text="Модель:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
            model_cb = ttk.Combobox(self.variables_frame, values=list(self.model_dict.keys()), state="readonly")
            model_cb.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
            model_cb.current(0)
            self._style_combobox(model_cb)
            self.model_var = model_cb
    
    def _style_combobox(self, combobox):
        """
        Дополнительная стилизация выпадающих списков для лучшей видимости.
        
        Args:
            combobox (ttk.Combobox): Виджет выпадающего списка
        """
        # Устанавливаем цвета напрямую для каждого combobox
        combobox.configure(
            foreground=DARK_THEME['text_light'],
            background=DARK_THEME['bg'],
            font=("Arial", 10, "bold")
        )
        
        # Для повышения контрастности выпадающих списков используем более светлый фон
        combobox.option_add('*TCombobox*Listbox.Background', DARK_THEME['bg_light'])
        combobox.option_add('*TCombobox*Listbox.Foreground', '#FFFFFF')  # Белый текст
        combobox.option_add('*TCombobox*Listbox.selectBackground', DARK_THEME['accent'])
        combobox.option_add('*TCombobox*Listbox.Font', ('Arial', 10, 'bold'))
        
        # Увеличиваем высоту списка для лучшей читаемости
        combobox.configure(height=10)
        
        # Добавим прямое изменение цвета поля для лучшей видимости
        try:
            # Находим поле ввода внутри combobox
            entry = combobox.nametowidget(combobox.winfo_children()[0])
            # Изменяем его цвета напрямую, если это возможно
            entry.configure(
                readonlybackground=DARK_THEME['bg_light'],  # Более светлый фон для контраста
                fg=DARK_THEME['text_light'],  # Светлый текст
                insertbackground=DARK_THEME['accent']  # Цвет курсора
            )
        except (KeyError, IndexError, AttributeError, tk.TclError):
            # Если не удалось найти или настроить entry, игнорируем
            pass
    
    def _reinitialize_canvas_and_toolbar(self):
        """
        Полностью пересоздает канвас и тулбар, сохраняя текущую фигуру.
        Используется, когда нужно восстановить связь между фигурой и элементами управления.
        """
        # Сохраняем текущую фигуру
        current_figure = self.figure
        
        # Удаляем существующий канвас и тулбар
        if hasattr(self, 'canvas') and self.canvas:
            for item in self.canvas_container.winfo_children():
                item.destroy()
        
        if hasattr(self, 'toolbar') and self.toolbar:
            for item in self.toolbar_frame.winfo_children():
                item.destroy()
        
        # Пересоздаем канвас с текущей фигурой
        self.canvas = FigureCanvasTkAgg(current_figure, self.canvas_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Пересоздаем тулбар с новым канвасом
        self.toolbar = RussianNavigationToolbar(self.canvas, self.toolbar_frame, theme=DARK_THEME)
        self.toolbar.update()
    
    def on_plot_type_change(self):
        """Обработчик изменения типа графика."""
        # Очищаем текущий график при смене типа, но не создаем новую фигуру
        self.figure.clear()
        
        # Обновляем элементы управления
        self.setup_variable_controls()
        
        # Полностью пересоздаем канвас и тулбар
        self._reinitialize_canvas_and_toolbar()
        
        # Перерисовываем канвас
        self.canvas.draw()
    
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
        
        # Очищаем текущую фигуру, но НЕ создаем новую
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
                
                # Создаем график, используя существующую фигуру
                create_scatter_plot(
                    x_data=x_data,
                    y_data=y_data,
                    feature_name=x_feature,
                    year_data=self.years,
                    highlight_year=year,
                    fig=self.figure
                )
                
                # Применяем темную тему к созданному графику
                self.apply_dark_theme_to_figure(self.figure)
                
                # Увеличиваем отступы для диаграммы рассеяния, чтобы были видны метки на нижней оси
                self.figure.subplots_adjust(bottom=0.23, left=0.12, right=0.92, top=0.85)
                
            elif plot_type == "multi_scatter":
                # Множественная диаграмма рассеяния
                selected_features = self.get_selected_features()
                
                if not selected_features:
                    messagebox.showwarning("Предупреждение", "Выберите хотя бы один признак для анализа")
                    return
                
                X_selected = self.df[selected_features]
                
                # Создаем базовую ось для графика
                ax = self.figure.add_subplot(111)
                
                # Вместо создания новой фигуры, используем существующую
                create_multi_scatter_plot(
                    X=X_selected,
                    y=y_data,
                    feature_names=selected_features,
                    target_name="ВВП",
                    fig=self.figure,
                    ax=ax
                )
                
                # Применяем темную тему к созданному графику
                self.apply_dark_theme_to_figure(self.figure)
                
                # Увеличиваем отступы для множественной диаграммы
                self.figure.subplots_adjust(bottom=0.23, left=0.12, right=0.92, top=0.85)
                
            elif plot_type == "3d":
                # 3D график
                x_feature = self.x_var.get()
                z_feature = self.z_var.get()
                
                if x_feature == z_feature:
                    messagebox.showwarning("Предупреждение", "Выберите разные признаки для осей X и Z")
                    return
                
                X_selected = self.df[[x_feature, z_feature]]
                
                # Создаем 3D ось
                ax = self.figure.add_subplot(111, projection='3d')
                
                # Используем существующую фигуру для 3D графика
                create_3d_plot(
                    X=X_selected,
                    y=y_data,
                    feature_names=[x_feature, z_feature],
                    target_name="ВВП",
                    years=self.years,
                    fig=self.figure,
                    ax=ax
                )
                
                # Применяем темную тему к созданному графику
                self.apply_dark_theme_to_figure(self.figure)
                
                # Корректируем положение 3D графика
                self.figure.subplots_adjust(bottom=0.23, left=0.12, right=0.92, top=0.85)
                
            elif plot_type == "heatmap":
                # Тепловая карта корреляций
                # Создаем DataFrame с признаками и целевой переменной
                data_for_corr = self.df[self.feature_names].copy()
                data_for_corr["ВВП"] = y_data
                
                # Создаем базовую ось
                ax = self.figure.add_subplot(111)
                
                # Создаем тепловую карту на существующей фигуре
                create_heatmap_plot(
                    data=data_for_corr,
                    title="Корреляции между экономическими показателями",
                    fig=self.figure,
                    ax=ax
                )
                
                # Применяем темную тему к созданному графику
                self.apply_dark_theme_to_figure(self.figure)
                
                # Корректируем положение тепловой карты
                self.figure.subplots_adjust(bottom=0.23, left=0.15, right=0.92, top=0.85)
                
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
                
                # Создаем базовую ось
                ax = self.figure.add_subplot(111)
                
                # Создаем график частичной зависимости на существующей фигуре
                create_partial_dependence_plot(
                    model=sklearn_model,
                    X=X_model,
                    feature_idx=feature_idx,
                    feature_name=x_feature,
                    target_name="ВВП",
                    fig=self.figure,
                    ax=ax
                )
                
                # Применяем темную тему к созданному графику
                self.apply_dark_theme_to_figure(self.figure)
                
                # Корректируем положение графика частичной зависимости
                self.figure.subplots_adjust(bottom=0.23, left=0.12, right=0.92, top=0.85)
            
            # Обновляем холст с корректным размером
            # Убедимся, что текущий размер фигуры соответствует размеру виджета
            w, h = self.graph_frame.winfo_width(), self.graph_frame.winfo_height()
            if w > 1 and h > 1:  # Если размеры уже известны
                # Устанавливаем горизонтальное соотношение сторон для автоматического масштабирования
                desired_width = w/90  # Уменьшаем ширину для лучшей видимости
                desired_height = desired_width / 1.8  # Увеличиваем относительную высоту (меньше 2:1)
                self.figure.set_size_inches(desired_width, desired_height)
                
                # Добавляем дополнительный отступ снизу для всех типов графиков
                self.figure.subplots_adjust(bottom=0.23, right=0.92)
            
            # Перерисовываем холст
            self.canvas.draw()
            
            # Пересоздаем все элементы для гарантии правильной работы
            self._reinitialize_canvas_and_toolbar()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при создании графика: {str(e)}")
            # Логируем исключение для отладки
            import logging
            logging.exception("Ошибка при создании графика:")
    
    def apply_dark_theme_to_figure(self, fig):
        """Применяет темную тему к уже созданной фигуре matplotlib"""
        fig.set_facecolor(DARK_THEME['primary'])
        
        # Применяем темную тему к каждой оси
        for ax in fig.get_axes():
            ax.set_facecolor(DARK_THEME['bg'])
            
            # Обновляем цвета для границ
            for spine in ax.spines.values():
                spine.set_color(DARK_THEME['neutral'])
            
            # Обновляем цвета для меток осей, делений и заголовка
            ax.tick_params(colors=DARK_THEME['neutral'])
            
            # Обновляем цвета для текстовых элементов
            if ax.get_xlabel():
                ax.xaxis.label.set_color(DARK_THEME['neutral'])
            if ax.get_ylabel():
                ax.yaxis.label.set_color(DARK_THEME['neutral'])
            if ax.get_title():
                ax.title.set_color(DARK_THEME['text_light'])
            
            # Обновляем цвета для сетки
            ax.grid(color=DARK_THEME['bg_light'], linestyle='--', alpha=0.3)
            
            # Обновляем цвета для текстовых аннотаций
            for child in ax.get_children():
                if isinstance(child, plt.Text):
                    child.set_color(DARK_THEME['neutral'])
            
            # Обновляем цвета для легенды, если она есть
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(DARK_THEME['bg'])
                legend.get_frame().set_edgecolor(DARK_THEME['neutral'])
                for text in legend.get_texts():
                    text.set_color(DARK_THEME['neutral']) 