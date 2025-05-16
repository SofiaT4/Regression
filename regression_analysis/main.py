#!/usr/bin/env python
"""
Главная точка входа для приложения регрессионного анализа.

Инициализирует приложение и запускает основной интерфейс,
обеспечивая корректную загрузку всех необходимых модулей.
"""

import tkinter as tk
import sys
import os
import logging
from datetime import datetime

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настраиваем логирование только в консоль (без файлов)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импортируем основной класс приложения из модуля ui
try:
    from ui.regression_app import RegressionApp
    # Импортируем функцию для применения темы
    from ui.components.theme_manager import apply_theme
    logger.info("Основной класс приложения успешно импортирован")
except ImportError as e:
    logger.error(f"Ошибка при импорте основного класса приложения: {e}")
    raise

# --- Monkey-patch для русификации окна настройки subplot ---
def patch_subplot_tool():
    import matplotlib
    from matplotlib.widgets import SubplotTool
    import tkinter as tk
    from tkinter import ttk

    orig_init = SubplotTool.__init__

    def russian_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        toolfig = getattr(self, 'toolfig', None)
        if toolfig is None:
            toolfig = getattr(self, '_toolfig', None)
        if toolfig is not None:
            toolfig.wm_title("Настройка подграфиков")
            for widget in toolfig.winfo_children():
                if isinstance(widget, tk.Label) and "Click on slider" in widget.cget("text"):
                    widget.config(text="Щелкните на ползунке для настройки параметров графика")
            # Стилизуем кнопку Reset (ищем рекурсивно)
            try:
                from ui.components.theme_manager import DARK_THEME
                def style_reset_button(widget):
                    # Для tk.Button
                    if isinstance(widget, tk.Button) and widget.cget("text") == "Reset":
                        widget.config(
                            bg=DARK_THEME['error'],
                            fg=DARK_THEME['text_light'],
                            activebackground=DARK_THEME['bg_light'],
                            activeforeground=DARK_THEME['neutral']
                        )
                    # Для ttk.Button
                    if isinstance(widget, ttk.Button) and widget.cget("text") == "Reset":
                        style = ttk.Style()
                        style.configure("Red.TButton",
                            background=DARK_THEME['error'],
                            foreground=DARK_THEME['text_light']
                        )
                        widget.config(style="Red.TButton")
                    # Рекурсивно для всех дочерних
                    if hasattr(widget, 'winfo_children'):
                        for child in widget.winfo_children():
                            style_reset_button(child)
                style_reset_button(toolfig)
            except Exception as e:
                pass

    SubplotTool.__init__ = russian_init

patch_subplot_tool()
# --- Конец monkey-patch ---

def main():
    """
    Основная функция, запускающая приложение регрессионного анализа.
    
    Создает корневое окно Tkinter, инициализирует основной класс приложения
    и запускает главный цикл обработки событий.
    """
    try:
        logger.info("Запуск приложения регрессионного анализа")
        
        # Создаем корневое окно Tkinter
        root = tk.Tk()
        root.title("Регрессионный анализ")
        
        # Запуск приложения в полноэкранном режиме
        root.attributes('-fullscreen', True)
        
        # Добавляем возможность выхода из полноэкранного режима по клавише Escape
        def toggle_fullscreen(event=None):
            is_fullscreen = root.attributes('-fullscreen')
            root.attributes('-fullscreen', not is_fullscreen)
            
        root.bind('<Escape>', toggle_fullscreen)
        logger.info("Приложение запущено в полноэкранном режиме")
        
        # Применяем темную тему ко всему приложению
        apply_theme(root)
        logger.info("Применен темный стиль оформления приложения")
        
        # Инициализируем приложение
        app = RegressionApp(root)
        logger.info("Приложение инициализировано")
        
        # Функция для показа модального окна подтверждения выхода
        def confirm_exit():
            from ui.components.theme_manager import DARK_THEME
            confirm_dialog = tk.Toplevel(root)
            confirm_dialog.title("Подтверждение выхода")
            confirm_dialog.attributes('-topmost', True)
            confirm_dialog.geometry("350x150")
            confirm_dialog.resizable(False, False)
            apply_theme(confirm_dialog)
            
            # Центрируем диалог на экране
            from ui.components.ui_helpers import center_window
            center_window(confirm_dialog, 350, 150)
            
            # Делаем диалог модальным
            confirm_dialog.grab_set()
            confirm_dialog.focus_set()
            
            # Содержимое диалога
            tk.Label(
                confirm_dialog,
                text="Вы действительно хотите выйти?",
                font=("Arial", 12, "bold"),
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral']
            ).pack(pady=20)
            
            # Фрейм для кнопок
            button_frame = tk.Frame(confirm_dialog, bg=DARK_THEME['primary'])
            button_frame.pack(pady=15)
            
            # Кнопка "Да"
            tk.Button(
                button_frame,
                text="Да",
                width=10,
                font=("Arial", 11),
                bg=DARK_THEME['error'],
                fg=DARK_THEME['text_light'],
                activebackground=DARK_THEME['bg_light'],
                activeforeground=DARK_THEME['neutral'],
                command=lambda: sys.exit(0)
            ).pack(side=tk.LEFT, padx=10)
            
            # Кнопка "Нет"
            tk.Button(
                button_frame,
                text="Нет",
                width=10,
                font=("Arial", 11),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light'],
                command=confirm_dialog.destroy
            ).pack(side=tk.LEFT, padx=10)
            
        # Добавляем кнопку выхода в правый верхний угол
        from ui.components.theme_manager import DARK_THEME
        exit_button = tk.Button(
            root, 
            text="✕", 
            font=("Arial", 14, "bold"),
            bg=DARK_THEME['error'],
            fg=DARK_THEME['text_light'],
            activebackground=DARK_THEME['bg_light'],
            activeforeground=DARK_THEME['neutral'],
            command=confirm_exit,
            width=3,
            height=1
        )
        exit_button.place(x=root.winfo_screenwidth() - 50, y=10)
        
        # Устанавливаем обработчик для корректного завершения приложения
        def on_closing():
            logger.info("Завершение работы приложения")
            confirm_exit()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Запускаем главный цикл обработки событий
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске приложения: {e}", exc_info=True)
        
        # Показываем сообщение об ошибке
        try:
            error_window = tk.Tk()
            error_window.title("Ошибка запуска")
            error_window.geometry("500x200")
            
            # Применяем стиль к окну ошибки
            apply_theme(error_window)
            
            tk.Label(
                error_window, 
                text="Произошла ошибка при запуске приложения:", 
                font=("Arial", 12, "bold")
            ).pack(pady=10)
            
            tk.Label(
                error_window, 
                text=str(e), 
                font=("Arial", 10),
                wraplength=450
            ).pack(pady=10)
            
            tk.Button(
                error_window, 
                text="Закрыть", 
                command=error_window.destroy
            ).pack(pady=10)
            
            error_window.mainloop()
        except:
            # Если не удалось показать графическое сообщение об ошибке,
            # выводим в консоль
            print(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()