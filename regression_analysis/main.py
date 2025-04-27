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

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"regression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
        
        # Применяем темную тему ко всему приложению
        apply_theme(root)
        logger.info("Применен темный стиль оформления приложения")
        
        # Инициализируем приложение
        app = RegressionApp(root)
        logger.info("Приложение инициализировано")
        
        # Устанавливаем обработчик для корректного завершения приложения
        def on_closing():
            logger.info("Завершение работы приложения")
            root.destroy()
        
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