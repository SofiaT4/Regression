"""
Модуль для отображения графиков и диаграмм регрессионного анализа.

Содержит класс для создания окна просмотра различных типов графиков,
с возможностью сохранения и интерактивного взаимодействия.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from utils.visualization.graph_manager import create_graph
from ui.components.ui_helpers import center_window, RussianNavigationToolbar

class GraphViewer:
    """
    Класс для отображения графиков в отдельном окне.
    Поддерживает интерактивное взаимодействие и сохранение графиков.
    """
    def __init__(self, parent: tk.Tk, graph_index: int, df: pd.DataFrame, 
            X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            model: Any, y_pred: np.ndarray, model_type: str):
        """
        Инициализация окна просмотра графика.
        
        Parameters:
        parent (tk.Tk or tk.Toplevel): Родительское окно
        graph_index (int): Индекс типа графика (0-3)
        df (pandas.DataFrame): Исходные данные
        X (pandas.DataFrame or numpy.ndarray): Признаки модели
        y (pandas.Series or numpy.ndarray): Целевая переменная
        model (sklearn.linear_model): Обученная модель регрессии
        y_pred (numpy.ndarray): Предсказанные значения
        model_type (str): Тип модели ('all_groups', 'unemployed', 'combined')
        """
        # Подавляем предупреждения о неверных именах признаков при создании графиков
        warnings.filterwarnings("ignore", category=UserWarning, 
                            message="X does not have valid feature names")
        
        # Импортируем компоненты темы
        from ui.components.theme_manager import DARK_THEME, apply_theme, get_chart_colors
        
        self.window = tk.Toplevel(parent)
        self.window.title(f"График {graph_index+1}")
        self.window.geometry("800x600")
        center_window(self.window, 800, 600)
        
        # Применяем тему к окну
        apply_theme(self.window)
        
        self.graph_index = graph_index
        self.df = df
        self.X = X
        self.y = y
        self.model = model
        self.y_pred = y_pred
        self.model_type = model_type
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса окна с графиком."""
        # Импортируем компоненты темы
        from ui.components.theme_manager import DARK_THEME, apply_chart_style
        
        # Создаем фрейм для графика
        graph_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Получаем описательный заголовок графика
        graph_titles = [
            "Фактический и прогнозируемый ВВП",
            "Визуализация коэффициентов модели",
            "График остатков",
            "Динамика показателей (нормализованные значения)"
        ]
        
        # Устанавливаем заголовок окна
        if self.graph_index < len(graph_titles):
            self.window.title(f"{graph_titles[self.graph_index]}")
        
        # Создаем график используя функцию из graph_manager
        fig = create_graph(self.graph_index, self.df, self.X, self.y, self.model, self.y_pred, self.model_type)
        self.fig = fig
        
        # Применяем темный стиль к графику
        apply_chart_style(plt)
        
        # Устанавливаем цвет фона фигуры
        fig.patch.set_facecolor(DARK_THEME['primary'])
        
        # Для каждой оси применяем темный стиль
        for ax in fig.get_axes():
            ax.set_facecolor(DARK_THEME['bg'])
            ax.xaxis.label.set_color(DARK_THEME['neutral'])
            ax.yaxis.label.set_color(DARK_THEME['neutral'])
            ax.title.set_color(DARK_THEME['neutral'])
            ax.tick_params(colors=DARK_THEME['neutral'])
            for spine in ax.spines.values():
                spine.set_color(DARK_THEME['neutral'])
        
        # Встраиваем график в окно Tkinter
        canvas = FigureCanvasTkAgg(fig, graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Добавляем панель инструментов matplotlib с темным стилем
        toolbar_frame = tk.Frame(graph_frame, bg=DARK_THEME['primary'])
        toolbar_frame.pack(fill=tk.X)
        
        toolbar = RussianNavigationToolbar(canvas, toolbar_frame, theme=DARK_THEME)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Кнопки управления
        button_frame = tk.Frame(self.window, bg=DARK_THEME['primary'])
        button_frame.pack(fill=tk.X, pady=10)
        
        # Добавляем информацию о модели
        model_names = {
            'all_groups': "Модель от численности рабочих", 
            'unemployed': "Модель от безработицы", 
            'combined': "Комбинированная модель"
        }
        model_info = model_names.get(self.model_type, self.model_type)
        
        info_label = tk.Label(
            button_frame, 
            text=f"{model_info}", 
            font=("Arial", 10, "italic"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        info_label.pack(side=tk.TOP, pady=5)
        
        # Кнопка экспорта данных
        export_data_button = tk.Button(
            button_frame, 
            text="Экспортировать данные", 
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.export_plot_data
        )
        export_data_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка сохранения
        save_button = tk.Button(
            button_frame, 
            text="Сохранить график", 
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.save_current_plot
        )
        save_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка настройки (для некоторых типов графиков)
        if self.graph_index in [1, 3]:  # Для коэффициентов и показателей
            settings_button = tk.Button(
                button_frame, 
                text="Настройки графика", 
                font=("Arial", 12),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light'],
                command=self.show_plot_settings
            )
            settings_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка закрытия
        back_button = tk.Button(
            button_frame, 
            text="Закрыть", 
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.window.destroy
        )
        back_button.pack(side=tk.RIGHT, padx=10)
    
    def save_current_plot(self):
        """Сохраняет текущий график в файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG файлы", "*.png"), 
                ("JPEG файлы", "*.jpg"),
                ("PDF файлы", "*.pdf"),
                ("SVG файлы", "*.svg"),
                ("Все файлы", "*.*")
            ],
            title="Сохранить график"
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Сохранение", f"График сохранен в файл:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить график:\n{str(e)}")
    
    def export_plot_data(self):
        """Экспортирует данные графика в CSV файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV файлы", "*.csv"),
                ("Excel файлы", "*.xlsx"),
                ("Все файлы", "*.*")
            ],
            title="Экспортировать данные графика"
        )
        
        if not file_path:
            return
        
        try:
            # В зависимости от типа графика экспортируем соответствующие данные
            if self.graph_index == 0:  # Фактический и прогнозируемый ВВП
                # Создаем DataFrame с данными
                years = self.df['Год'] if 'Год' in self.df.columns else range(len(self.y))
                data = pd.DataFrame({
                    'Год': years,
                    'Фактический_ВВП': self.y,
                    'Прогнозируемый_ВВП': self.y_pred,
                    'Остатки': self.y - self.y_pred
                })
                
            elif self.graph_index == 1:  # Визуализация коэффициентов
                # Получаем коэффициенты (исключая константу)
                coefficients = self.model.coef_
                
                # Получаем имена признаков
                if hasattr(self.X, 'columns'):
                    feature_names = list(self.X.columns)
                else:
                    feature_names = [f'Признак {i+1}' for i in range(len(coefficients))]
                
                # Создаем DataFrame
                data = pd.DataFrame({
                    'Признак': feature_names,
                    'Коэффициент': coefficients
                })
                
                # Сортируем по абсолютному значению
                data['Абсолютное_значение'] = data['Коэффициент'].abs()
                data = data.sort_values('Абсолютное_значение', ascending=False)
                
            elif self.graph_index == 2:  # График остатков
                # Создаем DataFrame с остатками
                standardized_residuals = (self.y - self.y_pred) / np.std(self.y - self.y_pred)
                data = pd.DataFrame({
                    'Предсказанные_значения': self.y_pred,
                    'Остатки': self.y - self.y_pred,
                    'Стандартизированные_остатки': standardized_residuals
                })
                
            else:  # Динамика показателей
                # Создаем DataFrame с динамикой показателей
                years = self.df['Год'] if 'Год' in self.df.columns else range(len(self.y))
                data = pd.DataFrame({'Год': years, 'ВВП': self.y})
                
                # Добавляем признаки из X
                if hasattr(self.X, 'columns'):
                    for col in self.X.columns:
                        data[col] = self.X[col]
                else:
                    for i in range(self.X.shape[1]):
                        data[f'Признак_{i+1}'] = self.X[:, i]
            
            # Сохраняем данные в зависимости от расширения файла
            if file_path.lower().endswith('.xlsx'):
                data.to_excel(file_path, index=False)
            else:
                data.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')
            
            messagebox.showinfo("Экспорт", f"Данные графика экспортированы в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", f"Не удалось экспортировать данные:\n{str(e)}")
    
    def show_plot_settings(self):
        """Показывает диалог с настройками графика."""
        from ui.components.theme_manager import DARK_THEME, apply_theme
        
        # Создаем диалоговое окно
        settings_window = tk.Toplevel(self.window)
        settings_window.title("Настройки графика")
        settings_window.geometry("400x300")
        center_window(settings_window, 400, 300)
        settings_window.grab_set()  # Делаем окно модальным
        
        # Применяем тему к окну
        apply_theme(settings_window)
        
        # Настройки в зависимости от типа графика
        if self.graph_index == 1:  # Визуализация коэффициентов
            # Фрейм для настроек
            settings_frame = tk.Frame(settings_window, padx=20, pady=20, bg=DARK_THEME['primary'])
            settings_frame.pack(fill=tk.BOTH, expand=True)
            
            # Количество отображаемых коэффициентов
            num_coefs_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
            num_coefs_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(
                num_coefs_frame, 
                text="Количество коэффициентов:",
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral']
            ).pack(side=tk.LEFT, padx=5)
            
            # Создаем список возможных значений
            num_coefs_values = [5, 10, 15, 20, "Все"]
            num_coefs_var = tk.StringVar(value="10")
            
            # Настраиваем стиль для Combobox
            style = ttk.Style()
            style.configure(
                "TCombobox", 
                fieldbackground=DARK_THEME['bg'],
                background=DARK_THEME['bg_light'],
                foreground=DARK_THEME['neutral'],
                arrowcolor=DARK_THEME['neutral']
            )
            
            num_coefs_combobox = ttk.Combobox(
                num_coefs_frame, 
                textvariable=num_coefs_var,
                values=num_coefs_values,
                width=10,
                state="readonly"
            )
            num_coefs_combobox.pack(side=tk.LEFT, padx=10)
            
            # Ориентация графика
            orientation_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
            orientation_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(
                orientation_frame, 
                text="Ориентация:",
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral']
            ).pack(side=tk.LEFT, padx=5)
            
            orientation_var = tk.StringVar(value="Горизонтальная")
            
            horizontal_radio = tk.Radiobutton(
                orientation_frame, 
                text="Горизонтальная", 
                variable=orientation_var, 
                value="Горизонтальная",
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            horizontal_radio.pack(side=tk.LEFT, padx=10)
            
            vertical_radio = tk.Radiobutton(
                orientation_frame, 
                text="Вертикальная", 
                variable=orientation_var, 
                value="Вертикальная",
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            vertical_radio.pack(side=tk.LEFT, padx=10)
            
            # Показывать p-значения
            show_pvalues_var = tk.BooleanVar(value=True)
            show_pvalues_check = tk.Checkbutton(
                settings_frame,
                text="Показывать p-значения",
                variable=show_pvalues_var,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            show_pvalues_check.pack(anchor=tk.W, pady=10)
            
            # Выделять значимые коэффициенты
            highlight_significant_var = tk.BooleanVar(value=True)
            highlight_significant_check = tk.Checkbutton(
                settings_frame,
                text="Выделять значимые коэффициенты",
                variable=highlight_significant_var,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            highlight_significant_check.pack(anchor=tk.W, pady=10)
            
            # Кнопки
            button_frame = tk.Frame(settings_window, pady=10, bg=DARK_THEME['primary'])
            button_frame.pack(fill=tk.X)
            
            # Функция для применения настроек
            def apply_settings():
                try:
                    # Получаем настройки
                    num_coefs = num_coefs_var.get()
                    max_features = int(num_coefs) if num_coefs != "Все" else None
                    horizontal = orientation_var.get() == "Горизонтальная"
                    show_pvalues = show_pvalues_var.get()
                    highlight_significant = highlight_significant_var.get()
                    
                    # Закрываем окно настроек
                    settings_window.destroy()
                    
                    # Обновляем график
                    self.update_plot(
                        max_features=max_features,
                        horizontal=horizontal,
                        show_pvalues=show_pvalues,
                        highlight_significant=highlight_significant
                    )
                    
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось применить настройки:\n{str(e)}")
            
            # Кнопка применения настроек
            apply_button = tk.Button(
                button_frame,
                text="Применить",
                command=apply_settings,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            apply_button.pack(side=tk.RIGHT, padx=10)
            
            # Кнопка отмены
            cancel_button = tk.Button(
                button_frame,
                text="Отмена",
                command=settings_window.destroy,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            cancel_button.pack(side=tk.RIGHT, padx=10)
        
        elif self.graph_index == 3:  # Динамика показателей
            # Фрейм для настроек
            settings_frame = tk.Frame(settings_window, padx=20, pady=20, bg=DARK_THEME['bg'])
            settings_frame.pack(fill=tk.BOTH, expand=True)

            # Заголовок
            tk.Label(
                settings_frame,
                text="Выберите показатели для отображения:",
                font=("Arial", 11, "bold"),
                bg=DARK_THEME['bg'],
                fg=DARK_THEME['text']
            ).pack(anchor=tk.W, pady=5)

            # Получаем список доступных показателей
            if hasattr(self.X, 'columns'):
                features = list(self.X.columns)
            else:
                features = [f'Признак {i+1}' for i in range(self.X.shape[1])]

            # Создаем фрейм с прокруткой для флажков
            canvas = tk.Canvas(settings_frame, borderwidth=0, bg=DARK_THEME['bg'], highlightthickness=0)
            scrollbar = tk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg=DARK_THEME['bg'])

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Добавляем флажки для каждого признака
            feature_vars = {}

            for i, feature in enumerate(features):
                var = tk.BooleanVar(value=(i < 4))
                checkbutton = tk.Checkbutton(
                    scrollable_frame,
                    text=feature,
                    variable=var,
                    bg=DARK_THEME['bg'],
                    fg=DARK_THEME['text'],
                    selectcolor=DARK_THEME['bg_light'],
                    activebackground=DARK_THEME['bg'],
                    activeforeground=DARK_THEME['accent'],
                    font=("Arial", 10)
                )
                checkbutton.pack(anchor=tk.W, pady=2)
                feature_vars[feature] = var

            # Кнопки
            button_frame = tk.Frame(settings_window, pady=10, bg=DARK_THEME['bg'])
            button_frame.pack(fill=tk.X)

            # Функция для применения настроек
            def apply_settings():
                try:
                    selected_features = [f for f, var in feature_vars.items() if var.get()]
                    if not selected_features:
                        messagebox.showwarning("Предупреждение", "Выберите хотя бы один признак.")
                        return
                    settings_window.destroy()
                    self.update_plot(selected_features=selected_features)
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось применить настройки:\n{str(e)}")

            # Кнопка применения настроек
            apply_button = tk.Button(
                button_frame,
                text="Применить",
                command=apply_settings,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            apply_button.pack(side=tk.RIGHT, padx=10)

            # Кнопка отмены
            cancel_button = tk.Button(
                button_frame,
                text="Отмена",
                command=settings_window.destroy,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            cancel_button.pack(side=tk.RIGHT, padx=10)
    
    def update_plot(self, **kwargs):
        """
        Обновляет график с новыми настройками.
        
        Parameters:
        **kwargs: Дополнительные параметры для создания графика
        """
        from ui.components.theme_manager import DARK_THEME, apply_chart_style
        
        try:
            # Создаем новый график с переданными параметрами
            fig = create_graph(self.graph_index, self.df, self.X, self.y, self.model, self.y_pred, self.model_type, **kwargs)
            self.fig = fig
            
            # Применяем темный стиль к графику
            apply_chart_style(plt)
            
            # Обновляем темный стиль для нового графика
            fig.patch.set_facecolor(DARK_THEME['primary'])
            for ax in fig.get_axes():
                ax.set_facecolor(DARK_THEME['bg'])
                ax.xaxis.label.set_color(DARK_THEME['neutral'])
                ax.yaxis.label.set_color(DARK_THEME['neutral'])
                ax.title.set_color(DARK_THEME['neutral'])
                ax.tick_params(colors=DARK_THEME['neutral'])
                for spine in ax.spines.values():
                    spine.set_color(DARK_THEME['neutral'])
            
            # Обновляем график в окне
            for widget in self.window.winfo_children():
                if isinstance(widget, tk.Frame) and len(widget.winfo_children()) > 0:
                    for child in widget.winfo_children():
                        if isinstance(child, FigureCanvasTkAgg) or isinstance(child, RussianNavigationToolbar):
                            child.destroy()
            
            # Создаем фрейм для графика
            graph_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
            graph_frame.pack(fill=tk.BOTH, expand=True)
            
            # Встраиваем новый график
            canvas = FigureCanvasTkAgg(fig, graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем панель инструментов с темным стилем
            toolbar_frame = tk.Frame(graph_frame, bg=DARK_THEME['primary'])
            toolbar_frame.pack(fill=tk.X)
            
            toolbar = RussianNavigationToolbar(canvas, toolbar_frame, theme=DARK_THEME)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обновить график:\n{str(e)}")