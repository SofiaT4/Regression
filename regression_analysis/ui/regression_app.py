"""
Главный класс приложения регрессионного анализа.

Содержит логику основного окна приложения, управление загрузкой данных
и инициализацию процесса анализа.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.components.ui_helpers import center_window
from ui.dialogs.csv_dialog import CSVSettingsDialog
from ui.model_display import ModelDisplayFrame
from core.data_processor.data_loader import prepare_data
from core.models.regression_builder import build_models
from core.statistics.coefficient_stats import calculate_all_statistics

class RegressionApp:
    def __init__(self, root):
        """
        Инициализация основного приложения регрессионного анализа с увеличенными размерами окон.
        
        Parameters:
        root (tk.Tk): Корневое окно Tkinter
        """
        self.root = root
        self.root.title("Регрессионный анализ")
        self.root.geometry("500x300")  # Увеличиваем начальное окно
        self.root.resizable(False, False)
        
        # Применяем тему к корневому окну
        from ui.components.theme_manager import apply_theme
        apply_theme(self.root)
        
        center_window(self.root, 500, 300)
        
        self.setup_start_screen()
        
        # Инициализация переменных
        self.df = None
        self.models = None
        self.predictions = None
        self.stats_dict = None
        self.graph_window = None
        self.current_model = 'combined'
        
    def setup_start_screen(self):
        """Создает начальный экран с кнопкой загрузки файла, адаптированный для полноэкранного режима."""
        from ui.components.theme_manager import DARK_THEME
        
        self.start_frame = tk.Frame(self.root, bg=DARK_THEME['primary'])
        self.start_frame.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем кнопку выхода в правый верхний угол
        exit_button = tk.Button(
            self.root, 
            text="✕", 
            font=("Arial", 14, "bold"),
            bg=DARK_THEME['error'],
            fg=DARK_THEME['text_light'],
            activebackground=DARK_THEME['bg_light'],
            activeforeground=DARK_THEME['neutral'],
            command=self.confirm_exit,
            width=3,
            height=1
        )
        exit_button.place(x=self.root.winfo_screenwidth() - 50, y=10)
        
        # Создаем центрированный контейнер для содержимого
        content_frame = tk.Frame(self.start_frame, bg=DARK_THEME['primary'], padx=20, pady=20)
        content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        title_label = tk.Label(
            content_frame, 
            text="Регрессионный анализ", 
            font=("Arial", 24, "bold"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        title_label.pack(pady=30)
        
        load_button = tk.Button(
            content_frame, 
            text="Загрузить CSV файл", 
            command=self.load_data, 
            font=("Arial", 16), 
            width=25, 
            height=2,
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light']
        )
        load_button.pack(pady=30)
    
    def load_data(self):
        """Открывает диалог выбора файла и запускает процесс загрузки данных."""
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл с данными",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Открываем диалоговое окно для настройки параметров чтения CSV
            dialog = CSVSettingsDialog(self.root, file_path, self.on_csv_loaded)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при чтении файла:\n{str(e)}")
    
    def on_csv_loaded(self, df, age_groups=None):
        """
        Обработчик события после успешной загрузки CSV.
        
        Parameters:
        df (pandas.DataFrame): Загруженный и обработанный датафрейм
        age_groups (list): Обнаруженные возрастные группы
        """
        from ui.components.theme_manager import DARK_THEME
        
        self.df = df
        self.age_groups = age_groups
        
        # Показываем индикатор прогресса
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Обработка данных")
        progress_window.geometry("300x100")
        progress_window.configure(bg=DARK_THEME['primary'])
        
        center_window(progress_window, 300, 100)
        progress_window.grab_set()
        
        progress_label = tk.Label(
            progress_window, 
            text="Подготовка данных...",
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        progress_label.pack(pady=10)
        
        # Настраиваем стиль прогресс-бара
        style = ttk.Style()
        style.configure(
            "TProgressbar",
            troughcolor=DARK_THEME['primary'],
            background=DARK_THEME['accent'],
            bordercolor=DARK_THEME['border'],
            lightcolor=DARK_THEME['accent'],
            darkcolor=DARK_THEME['secondary']
        )
        
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate", style="TProgressbar")
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        progress_bar.start(10)
        
        # Обновляем интерфейс
        self.root.update()
        
        try:
            # Подготовка данных, построение моделей и расчет статистики
            X_all_groups, X_unemployed, X_combined, y = prepare_data(self.df, self.age_groups)
            self.models, self.predictions = build_models(X_all_groups, X_unemployed, X_combined, y)
            self.stats_dict = calculate_all_statistics(X_all_groups, X_unemployed, X_combined, y, self.models, self.predictions)
            
            # Сохраняем X и y для использования в графиках
            self.X_all_groups = X_all_groups
            self.X_unemployed = X_unemployed
            self.X_combined = X_combined
            self.y = y
            
            # Закрываем окно прогресса
            progress_window.destroy()
            
            # Показываем результаты
            self.show_results()
        except Exception as e:
            # В случае ошибки закрываем окно прогресса и показываем сообщение
            progress_window.destroy()
            messagebox.showerror("Ошибка", f"Ошибка при анализе данных:\n{str(e)}")
            
    def show_results(self):
        """Скрывает начальный экран и показывает результаты анализа в полноэкранном режиме."""
        self.start_frame.pack_forget()
        
        # Разрешаем изменение размера окна для результатов
        self.root.resizable(True, True)
        
        # Создаем экран отображения моделей и передаем все необходимые данные
        self.results_frame = ModelDisplayFrame(
            self.root, 
            self.df, 
            self.stats_dict, 
            self.models, 
            self.predictions,
            self.X_all_groups,
            self.X_unemployed,
            self.X_combined,
            self.y,
            self.age_groups,
            self.back_to_start
        )

    def back_to_start(self):
        """Возвращает к начальному экрану для загрузки нового файла, сохраняя полноэкранный режим."""
        if hasattr(self, 'results_frame'):
            # Check if the frame has a destroy method before calling it
            if hasattr(self.results_frame, 'destroy') and callable(self.results_frame.destroy):
                self.results_frame.destroy()
            # Delete the attribute to make sure we don't try to use it later
            delattr(self, 'results_frame')
        
        # Блокируем изменение размера для начального экрана
        self.root.resizable(False, False)
        
        # Перенастраиваем начальный экран с новым стилем
        self.setup_start_screen()

    def confirm_exit(self):
        """Отображает диалог подтверждения выхода и обрабатывает ответ."""
        response = messagebox.askyesno("Подтверждение выхода", "Вы действительно хотите выйти?")
        if response:
            self.root.destroy()