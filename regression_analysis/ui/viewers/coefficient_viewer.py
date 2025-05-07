"""
Модуль для отображения коэффициентов регрессионных моделей.

Содержит класс для создания окна просмотра коэффициентов модели,
их статистической значимости, доверительных интервалов и других показателей.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

from ui.components.ui_helpers import center_window, RussianNavigationToolbar

class CoefficientViewer:
    """
    Класс для отображения коэффициентов модели в отдельном окне.
    Особенно полезен для моделей с большим количеством коэффициентов.
    """
    def __init__(self, parent: tk.Tk, model_stats: Dict[str, Any]):
        """
        Инициализация окна просмотра коэффициентов.
        
        Parameters:
        parent (tk.Tk or tk.Toplevel): Родительское окно
        model_stats (dict): Статистические показатели модели
        """
        from ui.components.theme_manager import DARK_THEME, apply_theme, style_treeview_tags
        
        self.window = tk.Toplevel(parent)
        self.window.title("Коэффициенты модели")
        self.window.geometry("800x600")
        center_window(self.window, 800, 600)
        
        # Применяем тему к окну
        apply_theme(self.window)
        
        self.model_stats = model_stats
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса окна с коэффициентами."""
        from ui.components.theme_manager import DARK_THEME, style_treeview_tags
        
        # Верхний фрейм с информацией о модели
        info_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        info_frame.pack(fill=tk.X)
        
        # Отображаем основную информацию о модели
        r2_text = f"R² = {self.model_stats['r2']:.4f}, " \
                f"Adj.R² = {self.model_stats['adjusted_r2']:.4f}, " \
                f"Число наблюдений: {self.model_stats['observations']}"
        
        info_label = tk.Label(
            info_frame, 
            text=r2_text, 
            font=("Arial", 10, "italic"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        info_label.pack(pady=5)
        
        # Фрейм для таблицы коэффициентов
        frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем прокручиваемую таблицу
        table_frame = tk.Frame(frame, bg=DARK_THEME['primary'])
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Полосы прокрутки
        scroll_y = tk.Scrollbar(table_frame, orient="vertical", bg=DARK_THEME['bg_light'], troughcolor=DARK_THEME['primary'])
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scroll_x = tk.Scrollbar(table_frame, orient="horizontal", bg=DARK_THEME['bg_light'], troughcolor=DARK_THEME['primary'])
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Настраиваем стиль Treeview
        style = ttk.Style()
        style.configure(
            "Treeview", 
            background=DARK_THEME['bg'], 
            foreground=DARK_THEME['neutral'],
            fieldbackground=DARK_THEME['primary'],
            borderwidth=0
        )
        style.map(
            'Treeview', 
            background=[('selected', DARK_THEME['accent'])],
            foreground=[('selected', DARK_THEME['text_light'])]
        )
        
        # Заголовки Treeview
        style.configure(
            "Treeview.Heading", 
            background=DARK_THEME['bg_light'], 
            foreground=DARK_THEME['text_light'],
            relief="flat"
        )
        style.map(
            "Treeview.Heading",
            background=[('active', DARK_THEME['accent'])],
            foreground=[('active', DARK_THEME['text_light'])]
        )
        
        # Создаем таблицу с прокруткой
        self.coef_table = ttk.Treeview(
            table_frame, 
            columns=("1", "2", "3", "4", "5", "6", "7"), 
            show="headings", 
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        
        scroll_y.config(command=self.coef_table.yview)
        scroll_x.config(command=self.coef_table.xview)
        
        # Настраиваем заголовки и ширину столбцов
        self.coef_table.heading("1", text="Признак")
        self.coef_table.heading("2", text="Коэффициент")
        self.coef_table.heading("3", text="Стандартная ошибка")
        self.coef_table.heading("4", text="t-статистика")
        self.coef_table.heading("5", text="P-Значение")
        
        # Добавляем столбцы для доверительных интервалов, если они есть
        if 'lower_ci' in self.model_stats and 'upper_ci' in self.model_stats:
            self.coef_table.heading("6", text="Нижний 95% CI")
            self.coef_table.heading("7", text="Верхний 95% CI")
        else:
            # Скрываем неиспользуемые столбцы
            self.coef_table["displaycolumns"] = ("1", "2", "3", "4", "5")
        
        # Настраиваем ширину столбцов
        self.coef_table.column("1", width=250, minwidth=150)
        self.coef_table.column("2", width=120, minwidth=100)
        self.coef_table.column("3", width=150, minwidth=120)
        self.coef_table.column("4", width=120, minwidth=100)
        self.coef_table.column("5", width=120, minwidth=100)
        
        if 'lower_ci' in self.model_stats and 'upper_ci' in self.model_stats:
            self.coef_table.column("6", width=120, minwidth=100)
            self.coef_table.column("7", width=120, minwidth=100)
        
        self.coef_table.pack(fill=tk.BOTH, expand=True)
        
        # Настраиваем цвета для тегов значимости
        self.coef_table.tag_configure("significant", background="#1a662a", foreground="white") # Темно-зеленый
        self.coef_table.tag_configure("not_significant", background="#661a1a", foreground="white") # Темно-красный
        
        # Заполнение таблицы коэффициентов с выделением значимых коэффициентов
        self.populate_coefficient_table()
        
        # Фрейм с кнопками
        button_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        button_frame.pack(fill=tk.X)
        
        # Кнопка экспорта коэффициентов в CSV
        export_button = tk.Button(
            button_frame, 
            text="Экспортировать в CSV", 
            command=self.export_coefficients_to_csv,
            font=("Arial", 11),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light']
        )
        export_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка фильтрации значимых коэффициентов
        self.show_significant_var = tk.BooleanVar(value=False)
        filter_button = ttk.Checkbutton(
            button_frame, 
            text="Показать только значимые (p < 0.05)", 
            variable=self.show_significant_var,
            command=self.toggle_significant_only,
            style="TCheckbutton"
        )
        filter_button.pack(side=tk.LEFT, padx=20)
        
        # Визуализация коэффициентов
        if len(self.model_stats['coefficients']) > 1:
            visualize_button = tk.Button(
                button_frame,
                text="Визуализировать коэффициенты",
                command=self.visualize_coefficients,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            visualize_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка закрытия окна
        close_button = tk.Button(
            button_frame, 
            text="Закрыть", 
            command=self.window.destroy,
            font=("Arial", 11),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light']
        )
        close_button.pack(side=tk.RIGHT, padx=10)
    
    def populate_coefficient_table(self, significant_only: bool = False):
        """
        Заполняет таблицу коэффициентов.
        
        Parameters:
        significant_only (bool): Если True, отображаются только значимые коэффициенты (p < 0.05)
        """
        # Очищаем таблицу
        for item in self.coef_table.get_children():
            self.coef_table.delete(item)
        
        # Заполняем таблицу коэффициентов
        for i, name in enumerate(self.model_stats['feature_names']):
            if i < len(self.model_stats['coefficients']):
                # Проверяем значимость коэффициента (p < 0.05)
                is_significant = False
                if 'p_values' in self.model_stats and i < len(self.model_stats['p_values']):
                    is_significant = self.model_stats['p_values'][i] < 0.05
                
                # Если включен фильтр и коэффициент не значим, пропускаем
                if significant_only and not is_significant:
                    continue
                
                # Выделяем разными цветами значимые и незначимые коэффициенты
                tag = "significant" if is_significant else "not_significant"
                
                # Подготавливаем данные для строки
                row_values = [
                    name,
                    f"{self.model_stats['coefficients'][i]:.8f}"
                ]
                
                # Добавляем стандартную ошибку, если она есть
                if 'se_coefficients' in self.model_stats and i < len(self.model_stats['se_coefficients']):
                    row_values.append(f"{self.model_stats['se_coefficients'][i]:.8f}")
                else:
                    row_values.append("")
                
                # Добавляем t-статистику, если она есть
                if 't_values' in self.model_stats and i < len(self.model_stats['t_values']):
                    row_values.append(f"{self.model_stats['t_values'][i]:.4f}")
                else:
                    row_values.append("")
                
                # Добавляем p-значение, если оно есть
                if 'p_values' in self.model_stats and i < len(self.model_stats['p_values']):
                    p_value = self.model_stats['p_values'][i]
                    
                    # Форматируем p-значение и добавляем звездочки для обозначения значимости
                    if p_value < 0.001:
                        p_text = "< 0.001 ***"
                    elif p_value < 0.01:
                        p_text = f"{p_value:.4f} **"
                    elif p_value < 0.05:
                        p_text = f"{p_value:.4f} *"
                    else:
                        p_text = f"{p_value:.4f}"
                    
                    row_values.append(p_text)
                else:
                    row_values.append("")
                
                # Добавляем доверительные интервалы, если они есть
                if 'lower_ci' in self.model_stats and i < len(self.model_stats['lower_ci']):
                    row_values.append(f"{self.model_stats['lower_ci'][i]:.6f}")
                
                if 'upper_ci' in self.model_stats and i < len(self.model_stats['upper_ci']):
                    row_values.append(f"{self.model_stats['upper_ci'][i]:.6f}")
                
                # Добавляем строку в таблицу
                self.coef_table.insert(
                    "", "end", 
                    values=tuple(row_values),
                    tags=(tag,)
                )
        
        # Настраиваем цвета для тегов
        self.coef_table.tag_configure("significant", background="#e6ffe6")  # Светло-зеленый
        self.coef_table.tag_configure("not_significant", background="#ffe6e6")  # Светло-красный
    
    def toggle_significant_only(self):
        """Переключает отображение между всеми коэффициентами и только значимыми."""
        show_significant = self.show_significant_var.get()
        self.populate_coefficient_table(significant_only=show_significant)
    
    def export_coefficients_to_csv(self):
        """Экспортирует коэффициенты в CSV файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            title="Экспортировать коэффициенты в CSV"
        )
        
        if not file_path:
            return
            
        try:
            # Создаем DataFrame с коэффициентами
            coef_data = []
            for i, name in enumerate(self.model_stats['feature_names']):
                if i < len(self.model_stats['coefficients']):
                    row_data = {
                        'Признак': name,
                        'Коэффициент': self.model_stats['coefficients'][i]
                    }
                    
                    # Добавляем стандартную ошибку, если она есть
                    if 'se_coefficients' in self.model_stats and i < len(self.model_stats['se_coefficients']):
                        row_data['Стандартная_ошибка'] = self.model_stats['se_coefficients'][i]
                    
                    # Добавляем t-статистику, если она есть
                    if 't_values' in self.model_stats and i < len(self.model_stats['t_values']):
                        row_data['t-статистика'] = self.model_stats['t_values'][i]
                    
                    # Добавляем p-значение и значимость, если они есть
                    if 'p_values' in self.model_stats and i < len(self.model_stats['p_values']):
                        p_value = self.model_stats['p_values'][i]
                        row_data['P-значение'] = p_value
                        row_data['Значимость'] = 'Значим' if p_value < 0.05 else 'Не значим'
                    
                    # Добавляем доверительные интервалы, если они есть
                    if 'lower_ci' in self.model_stats and i < len(self.model_stats['lower_ci']):
                        row_data['Нижний_95%_CI'] = self.model_stats['lower_ci'][i]
                    
                    if 'upper_ci' in self.model_stats and i < len(self.model_stats['upper_ci']):
                        row_data['Верхний_95%_CI'] = self.model_stats['upper_ci'][i]
                    
                    coef_data.append(row_data)
            
            # Создаем DataFrame и сохраняем в CSV
            coef_df = pd.DataFrame(coef_data)
            coef_df.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')
            
            messagebox.showinfo("Экспорт", f"Коэффициенты успешно экспортированы в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте коэффициентов:\n{str(e)}")
    
    def visualize_coefficients(self):
        """Открывает окно с визуализацией коэффициентов."""
        from ui.components.theme_manager import DARK_THEME, apply_theme, get_chart_colors
        
        try:
            # Импортируем необходимые модули
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Создаем новое окно
            viz_window = tk.Toplevel(self.window)
            viz_window.title("Визуализация коэффициентов")
            viz_window.geometry("800x600")
            center_window(viz_window, 800, 600)
            
            # Применяем тему к окну
            apply_theme(viz_window)
            
            # Создаем фрейм для графика
            frame = tk.Frame(viz_window, bg=DARK_THEME['primary'])
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Создаем фигуру и ось
            fig = Figure(figsize=(10, 8), dpi=100, facecolor=DARK_THEME['primary'])
            ax = fig.add_subplot(111)
            
            # Применяем стиль к графику matplotlib
            ax.set_facecolor(DARK_THEME['bg'])
            ax.xaxis.label.set_color(DARK_THEME['neutral'])
            ax.yaxis.label.set_color(DARK_THEME['neutral'])
            ax.tick_params(colors=DARK_THEME['neutral'])
            ax.spines['bottom'].set_color(DARK_THEME['neutral'])
            ax.spines['left'].set_color(DARK_THEME['neutral'])
            
            # Получаем данные для визуализации
            coefficients = self.model_stats['coefficients'][1:]  # Исключаем константу
            feature_names = self.model_stats['feature_names'][1:]  # Исключаем константу
            
            # Сортируем коэффициенты по абсолютному значению
            sorted_indices = np.argsort(np.abs(coefficients))
            sorted_coeffs = coefficients[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            # Лимитируем количество отображаемых признаков
            max_features = 20
            if len(sorted_names) > max_features:
                sorted_names = sorted_names[-max_features:]
                sorted_coeffs = sorted_coeffs[-max_features:]
            
            # Определяем цвета в зависимости от знака коэффициента
            colors = [DARK_THEME['success'] if c > 0 else DARK_THEME['error'] for c in sorted_coeffs]
            
            # Создаем горизонтальную диаграмму
            bars = ax.barh(sorted_names, sorted_coeffs, color=colors, alpha=0.7)
            
            # Добавляем вертикальную линию в нуле
            ax.axvline(x=0, color=DARK_THEME['neutral'], linestyle='-', alpha=0.3)
            
            # Добавляем значения на концах столбцов
            for bar in bars:
                width = bar.get_width()
                if width >= 0:
                    x = width + 0.01
                    alignment = 'left'
                else:
                    x = width - 0.01
                    alignment = 'right'
                ax.text(x, bar.get_y() + bar.get_height()/2, 
                    f"{width:.4f}", ha=alignment, va='center', color=DARK_THEME['neutral'])
            
            # Добавляем подписи и заголовок
            ax.set_title('Коэффициенты регрессии', fontsize=14, pad=20, color=DARK_THEME['neutral'])
            ax.set_xlabel('Значение коэффициента', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
            
            # Удаляем лишние рамки
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Подгоняем макет
            fig.tight_layout()
            
            # Встраиваем график в окно
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем панель инструментов matplotlib (также с темным стилем)
            toolbar_frame = tk.Frame(frame, bg=DARK_THEME['primary'])
            toolbar_frame.pack(fill=tk.X)
            toolbar = RussianNavigationToolbar(canvas, toolbar_frame, theme=DARK_THEME)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Добавляем кнопку закрытия
            button_frame = tk.Frame(viz_window, pady=10, bg=DARK_THEME['primary'])
            button_frame.pack(fill=tk.X)
            
            close_button = tk.Button(
                button_frame, 
                text="Закрыть", 
                command=viz_window.destroy,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            close_button.pack(side=tk.RIGHT, padx=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании визуализации:\n{str(e)}")