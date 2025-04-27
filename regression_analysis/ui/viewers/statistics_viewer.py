"""
Модуль для просмотра статистических показателей регрессионных моделей.

Содержит класс для отображения статистики регрессии в виде таблиц 
и форматированного текста, включая R², F-статистику, t-тесты и другие показатели.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable

from ui.components.ui_helpers import center_window
from ui.dialogs.file_selector import save_file
from ui.components.theme_manager import DARK_THEME, apply_theme, style_treeview_tags

class StatisticsViewer:
    """
    Класс для отображения статистических показателей регрессионных моделей.
    
    Предоставляет интерфейс для просмотра, сравнения и экспорта
    статистических данных моделей регрессии в формате Excel.
    """
    def __init__(self, parent: tk.Tk, model_stats: Dict[str, Any], title: str = "Статистика регрессии"):
        """
        Инициализация окна просмотра статистики.
        
        Parameters:
        parent (tk.Tk): Родительское окно
        model_stats (Dict[str, Any]): Статистические показатели модели
        title (str): Заголовок окна
        """
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("800x600")
        center_window(self.window, 800, 600)
        
        # Применяем темную тему к окну
        apply_theme(self.window)
        
        self.model_stats = model_stats
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса окна со статистикой в стиле Excel."""
        # Верхний фрейм с информацией о модели
        info_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        info_frame.pack(fill=tk.X)
        
        # Отображаем основную информацию о модели
        r2_text = f"R² = {self.model_stats['r2']:.4f}, " \
                  f"Adj.R² = {self.model_stats['adjusted_r2']:.4f}, " \
                  f"Число наблюдений: {self.model_stats['observations']}"
        
        info_label = tk.Label(info_frame, text=r2_text, font=("Arial", 10, "italic"),
                            bg=DARK_THEME['primary'], fg=DARK_THEME['neutral'])
        info_label.pack(pady=5)
        
        # Создаем вкладки для различных типов статистики
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка "Регрессионная статистика"
        regression_frame = tk.Frame(self.notebook, bg=DARK_THEME['primary'])
        self.notebook.add(regression_frame, text="Регрессионная статистика")
        self.setup_regression_tab(regression_frame)
        
        # Вкладка "Дисперсионный анализ"
        anova_frame = tk.Frame(self.notebook, bg=DARK_THEME['primary'])
        self.notebook.add(anova_frame, text="Дисперсионный анализ")
        self.setup_anova_tab(anova_frame)
        
        # Вкладка "Коэффициенты"
        coef_frame = tk.Frame(self.notebook, bg=DARK_THEME['primary'])
        self.notebook.add(coef_frame, text="Коэффициенты")
        self.setup_coefficients_tab(coef_frame)
        
        # Кнопки в нижней части окна
        button_frame = tk.Frame(self.window, padx=10, pady=10, bg=DARK_THEME['primary'])
        button_frame.pack(fill=tk.X)
        
        # Кнопка экспорта статистики
        export_button = tk.Button(
            button_frame, 
            text="Экспортировать в CSV", 
            command=self.export_statistics_to_csv,
            font=("Arial", 11),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light']
        )
        export_button.pack(side=tk.LEFT, padx=10)
        
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
    
    def setup_regression_tab(self, parent_frame: tk.Frame):
        """
        Настройка вкладки с регрессионной статистикой в формате Excel.
        
        Parameters:
        parent_frame (tk.Frame): Родительский фрейм для вкладки
        """
        # Создаем таблицу для отображения основных показателей регрессии
        table_frame = tk.Frame(parent_frame, padx=10, pady=10, bg=DARK_THEME['primary'])
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Treeview для табличного отображения
        reg_table = ttk.Treeview(table_frame, columns=("1", "2"), show="headings", height=10)
        reg_table.heading("1", text="Показатель")
        reg_table.heading("2", text="Значение")
        reg_table.column("1", width=300)
        reg_table.column("2", width=300)
        
        # Применяем стили темной темы к Treeview
        style_treeview_tags(reg_table)
        
        # Заполняем таблицу данными
        reg_metrics = [
            ("Множественный R", f"{self.model_stats['multiple_r']:.8f}"),
            ("R-квадрат", f"{self.model_stats['r2']:.8f}"),
            ("Нормированный R-квадрат", f"{self.model_stats['adjusted_r2']:.8f}"),
            ("Стандартная ошибка", f"{self.model_stats['se_regression']:.8f}"),
            ("Наблюдения", f"{self.model_stats['observations']}"),
            ("F-статистика", f"{self.model_stats['f_statistic']:.8f}"),
            ("Значимость F", f"{self.model_stats['p_value_f']:.8f}")
        ]
        
        for metric in reg_metrics:
            reg_table.insert("", "end", values=metric)
        
        reg_table.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем полосы прокрутки
        scrollbar_y = ttk.Scrollbar(table_frame, orient="vertical", command=reg_table.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        reg_table.configure(yscrollcommand=scrollbar_y.set)
        
        # Добавляем информацию о значимости модели
        f_info_frame = tk.Frame(parent_frame, padx=10, pady=5, bg=DARK_THEME['primary'])
        f_info_frame.pack(fill=tk.X)
        
        p_value = self.model_stats['p_value_f']
        is_significant = p_value < 0.05
        
        significance_text = ""
        if p_value < 0.001:
            significance_text = "Модель высоко статистически значима (p < 0.001)"
        elif p_value < 0.01:
            significance_text = "Модель статистически значима (p < 0.01)"
        elif p_value < 0.05:
            significance_text = "Модель статистически значима (p < 0.05)"
        else:
            significance_text = "Модель статистически незначима (p ≥ 0.05)"
        
        significance_label = tk.Label(
            f_info_frame, 
            text=significance_text, 
            font=("Arial", 10, "italic"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['success'] if is_significant else DARK_THEME['error']
        )
        significance_label.pack(anchor=tk.W, pady=5)

    def setup_anova_tab(self, parent_frame: tk.Frame):
        """
        Настройка вкладки с дисперсионным анализом (ANOVA) в формате Excel.
        
        Parameters:
        parent_frame (tk.Frame): Родительский фрейм для вкладки
        """
        # Создаем заголовок
        title_label = tk.Label(parent_frame, text="Дисперсионный анализ", 
                             font=("Arial", 12, "bold"),
                             bg=DARK_THEME['primary'],
                             fg=DARK_THEME['neutral'])
        title_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Создаем фрейм для таблицы ANOVA
        anova_frame = tk.Frame(parent_frame, padx=10, pady=5, bg=DARK_THEME['primary'])
        anova_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Treeview для таблицы ANOVA
        columns = ("1", "2", "3", "4", "5", "6")
        anova_table = ttk.Treeview(anova_frame, columns=columns, show="headings", height=4)
        
        # Настраиваем заголовки и ширину столбцов
        anova_table.heading("1", text="")
        anova_table.heading("2", text="df")
        anova_table.heading("3", text="SS")
        anova_table.heading("4", text="MS")
        anova_table.heading("5", text="F")
        anova_table.heading("6", text="Значимость F")
        
        anova_table.column("1", width=100, anchor=tk.W)
        anova_table.column("2", width=70, anchor=tk.CENTER)
        anova_table.column("3", width=150, anchor=tk.CENTER)
        anova_table.column("4", width=150, anchor=tk.CENTER)
        anova_table.column("5", width=100, anchor=tk.CENTER)
        anova_table.column("6", width=150, anchor=tk.CENTER)
        
        # Применяем стили темной темы к Treeview
        style_treeview_tags(anova_table)
        
        # Заполняем таблицу ANOVA
        anova_table.insert("", "end", values=(
            "Регрессия", 
            f"{self.model_stats['df_regression']}", 
            f"{self.model_stats['ss_regression']:.8f}", 
            f"{self.model_stats['ms_regression']:.8f}", 
            f"{self.model_stats['f_statistic']:.8f}",
            f"{self.model_stats['p_value_f']:.8f}"
        ))
        
        anova_table.insert("", "end", values=(
            "Остаток", 
            f"{self.model_stats['df_residual']}", 
            f"{self.model_stats['ss_residual']:.8f}", 
            f"{self.model_stats['ms_residual']:.8f}", 
            "",
            ""
        ))
        
        anova_table.insert("", "end", values=(
            "Итого", 
            f"{self.model_stats['df_total']}", 
            f"{self.model_stats['ss_total']:.8f}", 
            "", 
            "",
            ""
        ))
        
        # Добавляем полосы прокрутки
        scrollbar_y = ttk.Scrollbar(anova_frame, orient="vertical", command=anova_table.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(anova_frame, orient="horizontal", command=anova_table.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        anova_table.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        anova_table.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем пояснения к таблице ANOVA
        explanation_frame = tk.Frame(parent_frame, padx=10, pady=5, bg=DARK_THEME['primary'])
        explanation_frame.pack(fill=tk.X)
        
        explanation_text = (
            "Таблица дисперсионного анализа (ANOVA) показывает разложение общей изменчивости зависимой переменной "
            "на объясненную моделью часть (Регрессия) и необъясненную часть (Остаток).\n\n"
            "• df - степени свободы\n"
            "• SS - сумма квадратов (Sum of Squares)\n"
            "• MS - средний квадрат (Mean Square = SS/df)\n"
            "• F - F-статистика (отношение MS регрессии к MS остатка)\n"
            "• Значимость F - p-значение для F-статистики"
        )
        
        explanation_label = tk.Label(
            explanation_frame, 
            text=explanation_text, 
            font=("Arial", 10), 
            justify=tk.LEFT,
            wraplength=700,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        explanation_label.pack(anchor=tk.W, pady=10)
        
        # Добавляем интерпретацию F-статистики
        f_val = self.model_stats['f_statistic']
        p_val = self.model_stats['p_value_f']
        
        interpretation_text = ""
        if p_val < 0.001:
            interpretation_text = (
                f"Значение F-статистики ({f_val:.4f}) с p-значением < 0.001 указывает на то, что "
                "модель ВЫСОКО статистически значима. Мы можем с большой уверенностью отвергнуть "
                "нулевую гипотезу о том, что все коэффициенты модели равны нулю."
            )
        elif p_val < 0.01:
            interpretation_text = (
                f"Значение F-статистики ({f_val:.4f}) с p-значением < 0.01 указывает на то, что "
                "модель статистически значима. Мы можем уверенно отвергнуть нулевую гипотезу о том, "
                "что все коэффициенты модели равны нулю."
            )
        elif p_val < 0.05:
            interpretation_text = (
                f"Значение F-статистики ({f_val:.4f}) с p-значением < 0.05 указывает на то, что "
                "модель статистически значима на стандартном уровне значимости. Мы можем отвергнуть "
                "нулевую гипотезу о том, что все коэффициенты модели равны нулю."
            )
        else:
            interpretation_text = (
                f"Значение F-статистики ({f_val:.4f}) с p-значением {p_val:.4f} указывает на то, что "
                "модель НЕ является статистически значимой. Мы не можем отвергнуть нулевую гипотезу "
                "о том, что все коэффициенты модели равны нулю."
            )
        
        interpretation_label = tk.Label(
            explanation_frame,
            text=interpretation_text,
            font=("Arial", 10, "italic"),
            justify=tk.LEFT,
            wraplength=700,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['success'] if p_val < 0.05 else DARK_THEME['error']
        )
        interpretation_label.pack(anchor=tk.W, pady=5)
    
    def setup_coefficients_tab(self, parent_frame: tk.Frame):
        """
        Настройка вкладки с коэффициентами регрессии в формате Excel.
        
        Parameters:
        parent_frame (tk.Frame): Родительский фрейм для вкладки
        """
        # Создаем таблицу коэффициентов
        table_frame = tk.Frame(parent_frame, padx=10, pady=10, bg=DARK_THEME['primary'])
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Treeview для табличного отображения коэффициентов точно как в Excel
        columns = ("1", "2", "3", "4", "5", "6", "7")
        coef_table = ttk.Treeview(
            table_frame, 
            columns=columns, 
            show="headings", 
            height=15
        )
        coef_table.heading("1", text="")
        coef_table.heading("2", text="Коэффициенты")
        coef_table.heading("3", text="Стандартная ошибка")
        coef_table.heading("4", text="t-статистика")
        coef_table.heading("5", text="P-Значение")
        coef_table.heading("6", text="Нижние 95%")
        coef_table.heading("7", text="Верхние 95%")
        
        # Настраиваем ширину столбцов
        coef_table.column("1", width=150, anchor=tk.W)
        coef_table.column("2", width=100, anchor=tk.CENTER)
        coef_table.column("3", width=150, anchor=tk.CENTER)
        coef_table.column("4", width=100, anchor=tk.CENTER)
        coef_table.column("5", width=100, anchor=tk.CENTER)
        coef_table.column("6", width=100, anchor=tk.CENTER)
        coef_table.column("7", width=100, anchor=tk.CENTER)
        
        # Настраиваем цвета для тегов
        coef_table.tag_configure("significant", background=DARK_THEME['success'])  # Темно-зеленый
        coef_table.tag_configure("not_significant", background=DARK_THEME['error'])  # Темно-красный
        
        # Заполняем таблицу данными
        # Используем Excel-форматированные имена, если доступны
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
        
        for i, name in enumerate(feature_names):
            if i < len(coefs):
                # Определяем, значим ли коэффициент
                is_significant = p_values[i] < 0.05
                tag = "significant" if is_significant else "not_significant"
                
                coef_table.insert(
                    "", "end", 
                    values=(
                        name, 
                        f"{coefs[i]:.8f}", 
                        f"{se_coefs[i]:.8f}", 
                        f"{t_values[i]:.8f}", 
                        f"{p_values[i]:.8f}",
                        f"{lower_ci[i]:.8f}",
                        f"{upper_ci[i]:.8f}"
                    ),
                    tags=(tag,)
                )
        
        # Добавляем полосы прокрутки
        scrollbar_y = ttk.Scrollbar(table_frame, orient="vertical", command=coef_table.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(table_frame, orient="horizontal", command=coef_table.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        coef_table.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        coef_table.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем фильтр для отображения только значимых коэффициентов
        filter_frame = tk.Frame(parent_frame, padx=10, pady=5, bg=DARK_THEME['primary'])
        filter_frame.pack(fill=tk.X)
        
        self.show_significant_var = tk.BooleanVar(value=False)
        filter_button = ttk.Checkbutton(
            filter_frame, 
            text="Показать только значимые коэффициенты (p < 0.05)", 
            variable=self.show_significant_var,
            command=lambda: self.filter_coefficients(coef_table),
            style="TCheckbutton"
        )
        filter_button.pack(side=tk.LEFT, padx=5)

        # Добавляем пояснение о значимости коэффициентов
        explanation_frame = tk.Frame(parent_frame, padx=10, pady=5, bg=DARK_THEME['primary'])
        explanation_frame.pack(fill=tk.X)
        
        explanation_text = (
            "Коэффициенты, выделенные зеленым цветом, являются статистически значимыми (p < 0.05). "
            "Это означает, что соответствующие признаки вносят значимый вклад в модель.\n\n"
            "Звездочки обозначают уровень значимости: * p < 0.05, ** p < 0.01, *** p < 0.001"
        )
        
        explanation_label = tk.Label(
            explanation_frame,
            text=explanation_text,
            font=("Arial", 10),
            justify=tk.LEFT,
            wraplength=700,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        explanation_label.pack(anchor=tk.W, pady=5)
    
    def filter_coefficients(self, coef_table: ttk.Treeview):
        """
        Фильтрует коэффициенты в таблице по их значимости.
        
        Parameters:
        coef_table (ttk.Treeview): Таблица с коэффициентами
        """
        show_significant_only = self.show_significant_var.get()
        
        # Получаем все строки в таблице
        all_items = coef_table.get_children()
        
        for item in all_items:
            # Получаем теги элемента
            tags = coef_table.item(item, "tags")
            
            # Если включен фильтр и коэффициент не значим, скрываем его
            if show_significant_only and "not_significant" in tags:
                coef_table.detach(item)  # Скрываем элемент
            else:
                coef_table.reattach(item, "", "end")  # Показываем элемент
    
    def export_statistics_to_csv(self):
        """Экспортирует статистику в CSV файл."""
        # Показываем диалог сохранения файла
        file_path = save_file(
            self.window,
            title="Экспортировать статистику в CSV",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            defaultextension=".csv"
        )
        
        if not file_path:
            return
            
        try:
            # Создаем DataFrame с регрессионной статистикой
            reg_data = {
                'Показатель': [
                    'Множественный R', 
                    'R-квадрат', 
                    'Нормированный R-квадрат', 
                    'Стандартная ошибка', 
                    'Наблюдения', 
                    'F-статистика', 
                    'Значимость F'
                ],
                'Значение': [
                    self.model_stats['multiple_r'],
                    self.model_stats['r2'],
                    self.model_stats['adjusted_r2'],
                    self.model_stats['se_regression'],
                    self.model_stats['observations'],
                    self.model_stats['f_statistic'],
                    self.model_stats['p_value_f']
                ]
            }
            
            reg_df = pd.DataFrame(reg_data)
            
            # Создаем DataFrame с дисперсионным анализом
            anova_data = {
                '': ['Регрессия', 'Остаток', 'Итого'],
                'df': [
                    self.model_stats['df_regression'],
                    self.model_stats['df_residual'],
                    self.model_stats['df_total']
                ],
                'SS': [
                    self.model_stats['ss_regression'],
                    self.model_stats['ss_residual'],
                    self.model_stats['ss_total']
                ],
                'MS': [
                    self.model_stats['ms_regression'],
                    self.model_stats['ms_residual'],
                    ''
                ],
                'F': [
                    self.model_stats['f_statistic'],
                    '',
                    ''
                ],
                'Значимость F': [
                    self.model_stats['p_value_f'],
                    '',
                    ''
                ]
            }
            
            anova_df = pd.DataFrame(anova_data)
            
            # Создаем DataFrame с коэффициентами
            coef_data = []
            
            # Используем Excel-форматированные имена, если доступны
            if 'excel_feature_names' in self.model_stats:
                feature_names = self.model_stats['excel_feature_names']
            else:
                feature_names = self.model_stats['feature_names']
            
            for i, name in enumerate(feature_names):
                if i < len(self.model_stats['coefficients']):
                    coef_data.append({
                        'Переменная': name,
                        'Коэффициент': self.model_stats['coefficients'][i],
                        'Стандартная_ошибка': self.model_stats['se_coefficients'][i],
                        't-статистика': self.model_stats['t_values'][i],
                        'P-значение': self.model_stats['p_values'][i],
                        'Нижние_95%': self.model_stats['lower_ci'][i],
                        'Верхние_95%': self.model_stats['upper_ci'][i],
                        'Значимость': 'Значим' if self.model_stats['p_values'][i] < 0.05 else 'Не значим'
                    })
            
            coef_df = pd.DataFrame(coef_data)
            
            # Записываем все таблицы в один CSV файл
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                f.write("РЕГРЕССИОННАЯ СТАТИСТИКА\n")
                reg_df.to_csv(f, index=False, sep=';')
                
                f.write("\n\nДИСПЕРСИОННЫЙ АНАЛИЗ\n")
                anova_df.to_csv(f, index=False, sep=';')
                
                f.write("\n\nКОЭФФИЦИЕНТЫ\n")
                coef_df.to_csv(f, index=False, sep=';')
            
            messagebox.showinfo("Экспорт", f"Статистика успешно экспортирована в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте статистики:\n{str(e)}")