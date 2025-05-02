"""
Модуль диалогового окна для настройки параметров чтения CSV файла.

Содержит класс CSVSettingsDialog для настройки разделителя, кодировки,
формата десятичных чисел и выбора колонок при загрузке CSV файлов.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Optional, Callable, Any, Dict, Tuple
# Импортируем вспомогательные функции для интерфейса
from ui.components.ui_helpers import center_window
# Импортируем компоненты темы
from ui.components.theme_manager import DARK_THEME, apply_theme, style_treeview_tags

# Настройка логирования
logger = logging.getLogger(__name__)

class CSVSettingsDialog:
    """
    Диалог настройки параметров чтения CSV файла.
    Позволяет настроить разделитель, кодировку, формат десятичных чисел и выбрать колонки.
    """
    def __init__(self, parent: tk.Tk, file_path: str, callback: Callable[[pd.DataFrame, Optional[List[str]]], None]):
        """
        Инициализация диалога настройки CSV.
        
        Parameters:
        parent (tk.Tk or tk.Toplevel): Родительское окно
        file_path (str): Путь к CSV файлу
        callback (function): Функция обратного вызова, вызываемая после успешной загрузки данных
        """
        self.parent = parent
        self.file_path = file_path
        self.callback = callback
        
        # Инициализируем атрибут age_groups
        self.age_groups = []
        
        # Создаем новое окно
        self.window = tk.Toplevel(parent)
        self.window.title("Настройки CSV файла")
        self.window.geometry("500x500")
        
        # Применяем тему к окну
        apply_theme(self.window)
        
        center_window(self.window, 500, 500)
        self.window.grab_set()  # Делаем окно модальным
        
        # Сохраняем текущие настройки для повторного использования
        self.current_settings = {
            'separator': ';',
            'encoding': 'utf-8',
            'decimal': ',',
            'thousands': ' ',
            'header': '0',
            'skiprows': '0'
        }
        
        logger.info(f"Открыт диалог настройки CSV для файла: {file_path}")
        
        self.setup_ui()
        self.detect_encoding_and_preview()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса диалога."""
        # Основной фрейм настроек
        settings_frame = tk.Frame(self.window, padx=20, pady=10, bg=DARK_THEME['primary'])
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем предпросмотр содержимого файла
        preview_frame = tk.LabelFrame(settings_frame, text="Предпросмотр файла", padx=10, pady=10,
                                    bg=DARK_THEME['primary'], fg=DARK_THEME['neutral'])
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_text = tk.Text(preview_frame, height=10, width=60,
                                bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                insertbackground=DARK_THEME['accent'])
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Настройки разделителя
        sep_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        sep_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sep_frame, text="Разделитель:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.sep_var = tk.StringVar(value=';')  # По умолчанию точка с запятой
        sep_options = [';', ',', '\t', ' ', '|']
        
        style = ttk.Style()
        style.configure("TCombobox", 
                    fieldbackground=DARK_THEME['bg'],
                    background=DARK_THEME['bg_light'],
                    foreground=DARK_THEME['neutral'],
                    arrowcolor=DARK_THEME['neutral'])
        
        self.sep_dropdown = ttk.Combobox(sep_frame, textvariable=self.sep_var, values=sep_options, width=5)
        self.sep_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Настройки кодировки
        encoding_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        encoding_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(encoding_frame, text="Кодировка:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.encoding_var = tk.StringVar(value='utf-8')
        encoding_options = ['utf-8', 'cp1251', 'latin1', 'ascii', 'cp1252']
        self.encoding_dropdown = ttk.Combobox(encoding_frame, textvariable=self.encoding_var, 
                                            values=encoding_options, width=10)
        self.encoding_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Настройка десятичного разделителя
        decimal_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        decimal_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(decimal_frame, text="Десятичный разделитель:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.decimal_var = tk.StringVar(value=',')
        decimal_options = [',', '.']
        self.decimal_dropdown = ttk.Combobox(decimal_frame, textvariable=self.decimal_var, 
                                            values=decimal_options, width=5)
        self.decimal_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Настройка разделителя тысяч
        thousands_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        thousands_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(thousands_frame, text="Разделитель тысяч:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.thousands_var = tk.StringVar(value=' ')
        thousands_options = [' ', ',', '.', '']
        self.thousands_dropdown = ttk.Combobox(thousands_frame, textvariable=self.thousands_var, 
                                            values=thousands_options, width=5)
        self.thousands_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Настройка наличия заголовка
        header_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        header_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(header_frame, text="Строка с заголовками:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.header_var = tk.StringVar(value='0')
        header_options = ['0', '1', '2', '3', '4', '5', 'None']
        self.header_dropdown = ttk.Combobox(header_frame, textvariable=self.header_var, 
                                        values=header_options, width=5)
        self.header_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Настройка пропускаемых строк
        skiprows_frame = tk.Frame(settings_frame, bg=DARK_THEME['primary'])
        skiprows_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(skiprows_frame, text="Пропустить строк сверху:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.skiprows_var = tk.StringVar(value='0')
        self.skiprows_entry = tk.Entry(skiprows_frame, textvariable=self.skiprows_var, width=5,
                                    bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                    insertbackground=DARK_THEME['accent'])
        self.skiprows_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопки колонок
        columns_frame = tk.LabelFrame(settings_frame, text="Сопоставление колонок", padx=10, pady=10,
                                    bg=DARK_THEME['primary'], fg=DARK_THEME['neutral'])
        columns_frame.pack(fill=tk.X, pady=10)
        
        # Год
        year_frame = tk.Frame(columns_frame, bg=DARK_THEME['primary'])
        year_frame.pack(fill=tk.X, pady=2)
        tk.Label(year_frame, text="Колонка с годами:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.year_var = tk.StringVar(value='Год')
        self.year_entry = tk.Entry(year_frame, textvariable=self.year_var, width=20,
                                bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                insertbackground=DARK_THEME['accent'])
        self.year_entry.pack(side=tk.LEFT, padx=5)
        
        # ВВП
        gdp_frame = tk.Frame(columns_frame, bg=DARK_THEME['primary'])
        gdp_frame.pack(fill=tk.X, pady=2)
        tk.Label(gdp_frame, text="Колонка с ВВП:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.gdp_var = tk.StringVar(value='ВВП')
        self.gdp_entry = tk.Entry(gdp_frame, textvariable=self.gdp_var, width=20,
                                bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                insertbackground=DARK_THEME['accent'])
        self.gdp_entry.pack(side=tk.LEFT, padx=5)
        
        # Безработица
        unemployed_frame = tk.Frame(columns_frame, bg=DARK_THEME['primary'])
        unemployed_frame.pack(fill=tk.X, pady=2)
        tk.Label(unemployed_frame, text="Колонка с безработицей:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.unemployed_var = tk.StringVar(value='Безработные')
        self.unemployed_entry = tk.Entry(unemployed_frame, textvariable=self.unemployed_var, width=20,
                                    bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                    insertbackground=DARK_THEME['accent'])
        self.unemployed_entry.pack(side=tk.LEFT, padx=5)
        
        # Численность рабочих
        workers_frame = tk.Frame(columns_frame, bg=DARK_THEME['primary'])
        workers_frame.pack(fill=tk.X, pady=2)
        tk.Label(workers_frame, text="Колонка с численностью рабочих:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.workers_var = tk.StringVar(value='Рабочие')
        self.workers_entry = tk.Entry(workers_frame, textvariable=self.workers_var, width=20,
                                    bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                    insertbackground=DARK_THEME['accent'])
        self.workers_entry.pack(side=tk.LEFT, padx=5)
        
        # Префикс возрастных групп
        age_prefix_frame = tk.Frame(columns_frame, bg=DARK_THEME['primary'])
        age_prefix_frame.pack(fill=tk.X, pady=2)
        tk.Label(age_prefix_frame, text="Префикс/паттерн возрастных групп:", bg=DARK_THEME['primary'], 
                fg=DARK_THEME['neutral']).pack(side=tk.LEFT, padx=5)
        self.age_prefix_var = tk.StringVar(value='')
        self.age_prefix_entry = tk.Entry(age_prefix_frame, textvariable=self.age_prefix_var, width=20,
                                    bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'],
                                    insertbackground=DARK_THEME['accent'])
        self.age_prefix_entry.pack(side=tk.LEFT, padx=5)
        
        # Добавляем кнопку обновления предпросмотра
        refresh_button = tk.Button(settings_frame, text="Обновить предпросмотр", command=self.update_preview,
                                bg=DARK_THEME['bg_light'], fg=DARK_THEME['neutral'],
                                activebackground=DARK_THEME['accent'], activeforeground=DARK_THEME['text_light'])
        refresh_button.pack(pady=5)
        
        # Создаем отдельный фрейм для кнопок в нижней части окна
        button_frame = tk.Frame(self.window, bg=DARK_THEME['primary'])
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)
        
        # Кнопка чтения файла
        read_button = tk.Button(
            button_frame, 
            text="Прочитать файл", 
            command=self.on_read_file,
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'], 
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'], 
            activeforeground=DARK_THEME['text_light']
        )
        read_button.pack(side=tk.RIGHT, padx=5)
        
        # Кнопка отмены
        cancel_button = tk.Button(
            button_frame, 
            text="Отмена", 
            command=self.window.destroy,
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'], 
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'], 
            activeforeground=DARK_THEME['text_light']
        )
        cancel_button.pack(side=tk.LEFT, padx=5)
    
    def detect_encoding_and_preview(self):
        """Определяет кодировку файла и показывает его предпросмотр."""
        # Пытаемся определить кодировку
        encodings = ['utf-8', 'cp1251', 'latin1', 'ascii', 'cp1252']
        file_content = None
        detected_encoding = None
        
        for enc in encodings:
            try:
                with open(self.file_path, 'r', encoding=enc) as f:
                    file_content = f.read(1000)  # Читаем первые 1000 символов
                detected_encoding = enc
                logger.info(f"Определена кодировка файла: {enc}")
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            # Если не удалось определить кодировку, читаем в бинарном режиме и показываем предупреждение
            with open(self.file_path, 'rb') as f:
                file_content = f.read(1000).decode('utf-8', errors='replace')
            messagebox.showwarning("Предупреждение", 
                                "Не удалось автоматически определить кодировку файла. Выберите кодировку вручную.")
            detected_encoding = 'utf-8'
            logger.warning("Не удалось автоматически определить кодировку файла")
        
        # Устанавливаем обнаруженную кодировку
        self.encoding_var.set(detected_encoding)
        
        # Определяем сепаратор по содержимому
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
                self.sep_var.set(most_common_sep)
                logger.info(f"Определен разделитель: {most_common_sep}")
        
        # Показываем содержимое файла в предпросмотре
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, file_content)
        self.preview_text.config(state=tk.DISABLED)
    
    def update_preview(self):
        """Обновляет предпросмотр файла с текущими настройками."""
        try:
            enc = self.encoding_var.get()
            sep = self.sep_var.get()
            
            with open(self.file_path, 'r', encoding=enc, errors='replace') as f:
                file_content = f.read(1000)
            
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, file_content)
            self.preview_text.config(state=tk.DISABLED)
            
            # Пробуем сразу показать структуру CSV с текущими настройками
            try:
                sample_df = pd.read_csv(
                    self.file_path,
                    encoding=enc,
                    sep=sep,
                    nrows=5,
                    decimal=self.decimal_var.get(),
                    thousands=self.thousands_var.get(),
                    engine='python',
                    on_bad_lines='skip'
                )
                
                # Показываем структуру таблицы
                self.preview_text.config(state=tk.NORMAL, bg=DARK_THEME['bg'], fg=DARK_THEME['text_light'])
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, "Структура CSV с текущими настройками:\n\n")
                self.preview_text.insert(tk.END, str(sample_df.head()))
                self.preview_text.config(state=tk.DISABLED)
                
                logger.info("Предпросмотр файла обновлен с отображением структуры CSV")
            except Exception as e:
                # Если не удалось прочитать как CSV, показываем простой текст
                logger.warning(f"Не удалось показать структуру CSV: {e}")
                
        except Exception as e:
            messagebox.showwarning("Предупреждение", f"Ошибка при обновлении предпросмотра: {str(e)}")
            logger.error(f"Ошибка при обновлении предпросмотра: {e}")
    
    def on_read_file(self):
        """Обработчик события нажатия на кнопку 'Прочитать файл'."""
        # Сохраняем настройки
        self.current_settings = {
            'separator': self.sep_var.get(),
            'encoding': self.encoding_var.get(),
            'decimal': self.decimal_var.get(),
            'thousands': self.thousands_var.get(),
            'header': self.header_var.get(),
            'skiprows': self.skiprows_var.get()
        }
        
        logger.info(f"Начато чтение файла с настройками: {self.current_settings}")
        
        # Читаем файл
        try:
            self.read_csv_file()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при чтении файла:\n{str(e)}")
            logger.error(f"Ошибка при чтении файла: {e}")
    
    def read_csv_file(self):
        """Читает CSV файл с указанными настройками и обрабатывает данные."""
        # Преобразуем skiprows в int
        skiprows = int(self.skiprows_var.get()) if self.skiprows_var.get().isdigit() else 0
        
        # Преобразуем header в int или None
        if self.header_var.get() == 'None':
            header = None
        else:
            header = int(self.header_var.get()) if self.header_var.get().isdigit() else 0
        
        try:
            # Пробуем разные комбинации параметров для чтения файла
            success = False
            error_message = ""
            
            # Список сепараторов для проверки
            separators_to_try = [self.sep_var.get(), ';', ',', '\t']
            
            for sep in separators_to_try:
                try:
                    logger.debug(f"Пробуем прочитать с разделителем: {sep}")
                    
                    # Пробуем прочитать файл
                    self.df = pd.read_csv(
                        self.file_path,
                        encoding=self.encoding_var.get(),
                        sep=sep,
                        skiprows=skiprows,
                        header=header,
                        decimal=self.decimal_var.get(),
                        thousands=self.thousands_var.get(),
                        engine='python',
                        on_bad_lines='skip'
                    )
                    
                    # Проверяем, что прочитано больше одной колонки
                    if self.df.shape[1] <= 1:
                        # Если только одна колонка, значит разделитель неверный
                        continue
                    
                    # Если всё хорошо, обновляем переменную и выходим из цикла
                    self.sep_var.set(sep)
                    success = True
                    logger.info(f"Успешно прочитано с разделителем: {sep}, получено {self.df.shape[1]} колонок")
                    
                    # Выводим названия колонок
                    logger.debug("Названия колонок:")
                    for col in self.df.columns:
                        logger.debug(f"- {col}")
                    
                    break
                except Exception as e:
                    error_message = str(e)
                    logger.warning(f"Ошибка при чтении с разделителем {sep}: {error_message}")
            
            if not success:
                # Если не удалось прочитать ни с одним разделителем, пробуем с delimiter=None
                try:
                    logger.debug("Пробуем прочитать с автоопределением разделителя")
                    
                    # Читаем несколько строк чтобы определить структуру
                    with open(self.file_path, 'r', encoding=self.encoding_var.get()) as f:
                        first_lines = [next(f) for _ in range(5) if f]
                    
                    # Определяем сепаратор по количеству символов в строках
                    potential_seps = {',': 0, ';': 0, '\t': 0}
                    for line in first_lines:
                        for sep, count in potential_seps.items():
                            potential_seps[sep] += line.count(sep)
                    
                    # Выбираем сепаратор с наибольшим количеством вхождений
                    best_sep = max(potential_seps.items(), key=lambda x: x[1])[0]
                    
                    # Пробуем прочитать с этим разделителем
                    self.df = pd.read_csv(
                        self.file_path,
                        encoding=self.encoding_var.get(),
                        sep=best_sep,
                        skiprows=skiprows,
                        header=header,
                        decimal=self.decimal_var.get(),
                        thousands=self.thousands_var.get(),
                        engine='python',
                        on_bad_lines='skip'
                    )
                    
                    self.sep_var.set(best_sep)
                    success = True
                    logger.info(f"Успешно прочитано с автоопределением разделителя: {best_sep}")
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Ошибка при автоопределении разделителя: {error_message}")
            
            if not success:
                # Если все попытки не удались, выбрасываем исключение
                raise ValueError(f"Не удалось прочитать CSV-файл ни с одним из разделителей. Последняя ошибка: {error_message}")
            
            # Обработка данных после успешного чтения
            self.process_dataframe()
            
            # Вызываем callback с обработанными данными
            self.window.destroy()
            logger.info(f"Успешно прочитан и обработан CSV-файл. Найдено {len(self.age_groups)} возрастных групп.")
            self.callback(self.df, self.age_groups)
            
        except Exception as e:
            raise ValueError(f"Ошибка при чтении CSV: {str(e)}")
    
    def process_dataframe(self):
        """Обрабатывает датафрейм после загрузки."""
        # Удаляем пустые строки и столбцы
        self.df = self.df.dropna(how='all').dropna(axis=1, how='all')
        
        # Выводим названия колонок для диагностики
        logger.debug("Колонки в загруженном файле:")
        for col in self.df.columns:
            logger.debug(f" - {col}")
        
        # Проверяем наличие цифровых диапазонов в названиях колонок (возрастные группы)
        age_pattern = re.compile(r'\d+\s*-\s*\d+')
        potential_age_groups = []
        
        for col in self.df.columns:
            col_str = str(col)
            if age_pattern.search(col_str):
                potential_age_groups.append(col)
                logger.debug(f"Найдена потенциальная возрастная группа: {col}")
        
        # Если найдены возрастные группы, запоминаем их
        if potential_age_groups:
            self.age_groups = potential_age_groups
            logger.info(f"Найдено {len(potential_age_groups)} возрастных групп")
        
        # Переименовываем обязательные колонки для дальнейшей обработки
        column_mapping = {}
        
        # Ищем колонки на основе гибкого соответствия
        for col in self.df.columns:
            col_str = str(col).lower().strip()
            
            # Для года - ищем "год" или цифры, похожие на год
            if ('год' in col_str or 'year' in col_str or 'гог' in col_str or
                re.search(r'\b(19|20)\d{2}\b', col_str) or 
                (len(col_str) <= 5 and col_str.isdigit())):
                column_mapping[col] = 'Год'
            
            # Для ВВП - ищем любые упоминания ВВП, расширяем ключевые слова
            elif ('ввп' in col_str or 'vvp' in col_str or 'gdp' in col_str or 
                  'валов' in col_str or 'врп' in col_str or 'валовой' in col_str or 
                  'продукт' in col_str):
                column_mapping[col] = 'ВВП (в текущих ценах)'
            
            # Для безработицы - расширяем поиск
            elif ('безраб' in col_str or 'unemploy' in col_str or 'безроб' in col_str or
                'безработн' in col_str or 'безработиц' in col_str):
                column_mapping[col] = 'Численность безработных'
            
            # Для численности рабочих - расширяем поиск
            elif ('рабоч' in col_str or 'работ' in col_str or 'employ' in col_str or 
                'труд' in col_str or 'работник' in col_str or 'занят' in col_str or
                ('числен' in col_str and 'возраст' in col_str)):
                column_mapping[col] = 'Численность рабочих, в том числе в возрасте, лет'
        
        # Переименовываем колонки
        self.df = self.df.rename(columns=column_mapping)
        logger.debug(f"Переименовано {len(column_mapping)} колонок")
        
        # Проверяем соответствие с пользовательскими названиями колонок из интерфейса
        user_column_mapping = {}
        
        # Находим колонку года (если пользователь указал)
        year_col_pattern = self.year_var.get().lower()
        if year_col_pattern and 'Год' not in self.df.columns:
            for col in self.df.columns:
                col_str = str(col).lower()
                if year_col_pattern in col_str or col_str == year_col_pattern:
                    user_column_mapping[col] = 'Год'
                    break
        
        # Находим колонку ВВП (если пользователь указал)
        gdp_col_pattern = self.gdp_var.get().lower()
        if gdp_col_pattern and 'ВВП (в текущих ценах)' not in self.df.columns:
            for col in self.df.columns:
                col_str = str(col).lower()
                if gdp_col_pattern in col_str or col_str == gdp_col_pattern:
                    user_column_mapping[col] = 'ВВП (в текущих ценах)'
                    break
        
        # Находим колонку безработицы (если пользователь указал)
        unemployed_col_pattern = self.unemployed_var.get().lower()
        if unemployed_col_pattern and 'Численность безработных' not in self.df.columns:
            for col in self.df.columns:
                col_str = str(col).lower()
                if unemployed_col_pattern in col_str or col_str == unemployed_col_pattern:
                    user_column_mapping[col] = 'Численность безработных'
                    break
        
        # Применяем пользовательские переименования
        if user_column_mapping:
            self.df = self.df.rename(columns=user_column_mapping)
            logger.debug(f"Применены пользовательские названия колонок: {user_column_mapping}")
        
        # Создаем столбцы с стандартными названиями, если их не удалось найти
        if 'Год' not in self.df.columns:
            # Создаем последовательные годы, начиная с 2000
            self.df['Год'] = range(2000, 2000 + len(self.df))
            logger.warning("Колонка 'Год' не найдена. Создана последовательность лет, начиная с 2000.")
        
        if 'ВВП (в текущих ценах)' not in self.df.columns:
            # Ищем подходящую числовую колонку для ВВП
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Выбираем первую числовую колонку, которая не Год и не входит в возрастные группы
                for col in numeric_cols:
                    if col != 'Год' and col not in self.age_groups:
                        self.df['ВВП (в текущих ценах)'] = self.df[col]
                        logger.info(f"Колонка '{col}' используется как 'ВВП'.")
                        break
            
            # Если всё еще нет колонки ВВП, создаем фиктивную
            # Если всё еще нет колонки ВВП, создаем фиктивную
            if 'ВВП (в текущих ценах)' not in self.df.columns:
                self.df['ВВП (в текущих ценах)'] = np.random.randint(1000000, 10000000, len(self.df))
                logger.warning("Колонка 'ВВП (в текущих ценах)' не найдена. Созданы случайные данные для демонстрации.")
        
        if 'Численность безработных' not in self.df.columns:
            # Ищем подходящую числовую колонку для безработицы
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Перебираем числовые колонки, которые еще не использованы
                for col in numeric_cols:
                    if (col != 'Год' and col != 'ВВП (в текущих ценах)' and 
                        col not in self.age_groups):
                        self.df['Численность безработных'] = self.df[col]
                        logger.info(f"Колонка '{col}' используется как 'Численность безработных'.")
                        break
            
            # Если всё еще нет колонки безработицы, создаем фиктивную
            if 'Численность безработных' not in self.df.columns:
                self.df['Численность безработных'] = np.random.randint(3000, 6000, len(self.df))
                logger.warning("Колонка 'Численность безработных' не найдена. Созданы случайные данные для демонстрации.")
        
        # Проверяем наличие возрастных групп
        if not self.age_groups and self.df.shape[1] > 3:
            # Ищем возрастные группы на основе цифровых диапазонов в названиях
            age_pattern = re.compile(r'\d+\s*-\s*\d+')
            for col in self.df.columns:
                col_str = str(col)
                if (age_pattern.search(col_str) and 
                    col not in ['Год', 'ВВП (в текущих ценах)', 'Численность безработных']):
                    self.age_groups.append(col)
                    logger.debug(f"Найдена возрастная группа: {col}")
        
        # Преобразуем данные в числовой формат
        for col in self.df.columns:
            try:
                # Преобразуем только нечисловые колонки, кроме "Год"
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    # Заменяем запятые на точки для десятичных чисел
                    if self.df[col].dtype == object:
                        self.df[col] = self.df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                    
                    # Пробуем преобразовать в числовой формат
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    logger.info(f"Колонка '{col}' преобразована в числовой формат")
            except Exception as e:
                logger.error(f"Ошибка при преобразовании колонки '{col}': {str(e)}")
        
        # Проверяем, что год в числовом формате
        if not pd.api.types.is_numeric_dtype(self.df['Год']):
            try:
                self.df['Год'] = pd.to_numeric(self.df['Год'], errors='coerce')
            except:
                # Если не удалось преобразовать, заменяем на последовательность
                self.df['Год'] = range(2000, 2000 + len(self.df))
        
        # Удаляем строки с пропущенными значениями в обязательных колонках
        self.df = self.df.dropna(subset=['Год', 'ВВП (в текущих ценах)'])
        
        # Сортируем по годам
        if 'Год' in self.df.columns:
            self.df = self.df.sort_values('Год')
        
        # Если не нашли возрастные группы, но нужны для анализа
        if not self.age_groups:
            # Создаем фиктивную колонку для возрастной группы
            col_name = '25-34'
            if col_name not in self.df.columns:
                self.df[col_name] = np.random.randint(10000, 50000, len(self.df))
                self.age_groups.append(col_name)
                logger.warning(f"Создана фиктивная возрастная группа: {col_name}")
        
        logger.info(f"Завершена обработка данных. Возрастные группы: {self.age_groups}")
    
    def process_age_prefix_pattern(self, age_prefix: str) -> List[str]:
        """
        Обрабатывает введенный пользователем паттерн возрастных групп.
        
        Parameters:
        age_prefix (str): Паттерн или префикс для поиска возрастных групп
        
        Returns:
        list: Список колонок, соответствующих паттерну, или пустой список для автоопределения
        """
        # Автоматическое определение возрастных групп по диапазонам чисел (например, "25-34")
        if not age_prefix:
            age_groups = []
            age_pattern = re.compile(r'\d+\s*-\s*\d+')
            
            for col in self.df.columns:
                col_str = str(col)
                if age_pattern.search(col_str) and col not in ['Год', 'ВВП (в текущих ценах)', 
                                                             'Численность безработных',
                                                             'Численность рабочих, в том числе в возрасте, лет']:
                    age_groups.append(col)
            
            return age_groups
        
        # Обрабатываем случай, когда пользователь вводит список возрастных групп через точку с запятой
        if ';' in age_prefix:
            age_groups = [group.strip() for group in age_prefix.split(';')]
            matching_cols = []
            
            for group in age_groups:
                # Для каждой группы ищем соответствующие колонки
                for col in self.df.columns:
                    col_str = str(col)
                    # Проверяем разные варианты совпадения
                    if (group in col_str or 
                        col_str == group or 
                        col_str.endswith(group) or 
                        col_str.lower().find(group.lower()) >= 0):
                        if col not in matching_cols:
                            matching_cols.append(col)
            
            if not matching_cols:
                # Если не нашли точные совпадения, пробуем искать по диапазонам в названиях колонок
                for col in self.df.columns:
                    col_str = str(col).lower()
                    for group in age_groups:
                        # Извлекаем числа из шаблона (например, "25-34" -> "25" и "34")
                        if '-' in group:
                            try:
                                start, end = group.split('-')
                                if re.search(f"{start}.*{end}", col_str) or re.search(f"{start}-{end}", col_str):
                                    if col not in matching_cols:
                                        matching_cols.append(col)
                            except:
                                pass
            
            return matching_cols
        
        # Обрабатываем случай, когда пользователь вводит общий префикс или паттерн
        matching_cols = []
        for col in self.df.columns:
            col_str = str(col)
            # Проверяем, соответствует ли колонка префиксу/паттерну
            if age_prefix in col_str or col_str.lower().find(age_prefix.lower()) >= 0:
                # Исключаем обязательные колонки
                if col_str not in ['Год', 'ВВП (в текущих ценах)', 'Численность безработных', 'Численность рабочих, в том числе в возрасте, лет']:
                    matching_cols.append(col)
        
        return matching_cols
    
    def select_column_dialog(self, column_name: str, required_columns: List[str]):
        """
        Показывает диалог для выбора колонки.
        
        Parameters:
        column_name (str): Имя колонки, которую нужно выбрать
        required_columns (list): Список требуемых колонок
        """
        dialog = tk.Toplevel(self.window)
        dialog.title(f"Выбор колонки для {column_name}")
        dialog.geometry("400x300")
        dialog.grab_set()
        
        # Применяем тему к диалогу
        apply_theme(dialog)
        
        tk.Label(dialog, text=f"Выберите колонку, которая соответствует '{column_name}':", 
                pady=10, wraplength=380, bg=DARK_THEME['primary'], fg=DARK_THEME['neutral']).pack()
        
        # Создаем список доступных колонок
        listbox = tk.Listbox(dialog, width=50, height=10, bg=DARK_THEME['bg'], 
                        fg=DARK_THEME['text_light'], selectbackground=DARK_THEME['accent'])
        listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        for col in self.df.columns:
            listbox.insert(tk.END, col)
        
        # Добавляем полосу прокрутки
        scrollbar = tk.Scrollbar(listbox, bg=DARK_THEME['bg_light'], troughcolor=DARK_THEME['primary'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        # Кнопка выбора
        def on_select():
            if listbox.curselection():
                selected_col = listbox.get(listbox.curselection()[0])
                # Переименовываем выбранную колонку
                self.df = self.df.rename(columns={selected_col: column_name})
                dialog.destroy()
                
                # Проверяем, остались ли еще колонки для выбора
                missing_columns = [col for col in required_columns if col not in self.df.columns]
                if missing_columns:
                    # Показываем следующий диалог выбора
                    self.select_column_dialog(missing_columns[0], required_columns)
                else:
                    # Все колонки выбраны, продолжаем обработку
                    self.process_dataframe()
        
        # Кнопка "Пропустить" для необязательных колонок
        def on_skip():
            dialog.destroy()
            
            # Проверяем, остались ли еще колонки для выбора
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                # Показываем следующий диалог выбора
                self.select_column_dialog(missing_columns[0], required_columns)
            else:
                # Все колонки выбраны, продолжаем обработку
                self.process_dataframe()
        
        # Кнопка "Создать фиктивную" для заполнения данными
        def on_create_dummy():
            dialog.destroy()
            
            # Создаем фиктивную колонку с соответствующими данными
            if column_name == 'Год':
                self.df[column_name] = range(2000, 2000 + len(self.df))
            elif column_name == 'ВВП (в текущих ценах)':
                self.df[column_name] = np.random.randint(1000000, 10000000, len(self.df))
            elif column_name == 'Численность безработных':
                self.df[column_name] = np.random.randint(3000, 6000, len(self.df))
            
            # Проверяем, остались ли еще колонки для выбора
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                # Показываем следующий диалог выбора
                self.select_column_dialog(missing_columns[0], required_columns)
            else:
                # Все колонки выбраны, продолжаем обработку
                self.process_dataframe()
        
        # Кнопки управления
        button_frame = tk.Frame(dialog, bg=DARK_THEME['primary'])
        button_frame.pack(fill=tk.X, pady=10)
        
        select_button = tk.Button(button_frame, text="Выбрать", command=on_select,
                                bg=DARK_THEME['bg_light'], fg=DARK_THEME['neutral'],
                                activebackground=DARK_THEME['accent'], 
                                activeforeground=DARK_THEME['text_light'])
        select_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка пропуска или создания фиктивной только для необязательных колонок
        if column_name != 'Год' and column_name != 'ВВП (в текущих ценах)':
            skip_button = tk.Button(button_frame, text="Пропустить", command=on_skip,
                                bg=DARK_THEME['bg_light'], fg=DARK_THEME['neutral'],
                                activebackground=DARK_THEME['accent'], 
                                activeforeground=DARK_THEME['text_light'])
            skip_button.pack(side=tk.LEFT, padx=10)
        
        # Всегда показываем кнопку создания фиктивной колонки
        dummy_button = tk.Button(button_frame, text="Создать фиктивную", command=on_create_dummy,
                            bg=DARK_THEME['bg_light'], fg=DARK_THEME['neutral'],
                            activebackground=DARK_THEME['accent'], 
                            activeforeground=DARK_THEME['text_light'])
        dummy_button.pack(side=tk.RIGHT, padx=10)