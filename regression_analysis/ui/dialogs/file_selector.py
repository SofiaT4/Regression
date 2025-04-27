"""
Модуль с классами для выбора и сохранения файлов.

Содержит классы и функции для создания диалоговых окон выбора и сохранения файлов,
с дополнительными возможностями настройки и обработки выбора пользователя.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from typing import Optional, Callable, List, Tuple, Dict, Any
from ui.components.ui_helpers import center_window
from ui.components.theme_manager import DARK_THEME, apply_theme

class FileSelector:
    """
    Базовый класс для диалогов выбора файлов.
    
    Предоставляет общую функциональность для диалогов открытия и сохранения файлов,
    включая валидацию и обработку выбора пользователя.
    """
    def __init__(self, parent: tk.Tk, title: str = "Выбор файла", 
            initialdir: Optional[str] = None, filetypes: Optional[List[Tuple[str, str]]] = None):
        """
        Инициализация базового селектора файлов.
        
        Parameters:
        parent (tk.Tk): Родительское окно
        title (str): Заголовок диалогового окна
        initialdir (str, optional): Начальная директория
        filetypes (List[Tuple[str, str]], optional): Список поддерживаемых типов файлов
        """
        self.parent = parent
        self.title = title
        self.initialdir = initialdir or os.getcwd()
        self.filetypes = filetypes or [("Все файлы", "*.*")]
        self.selected_file = None
        self.callback = None
        
        # Применяем тему к родительскому окну
        apply_theme(parent)
    
    def show(self, callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """
        Показывает диалоговое окно выбора файла.
        
        Parameters:
        callback (Callable, optional): Функция обратного вызова, вызываемая после выбора файла
        
        Returns:
        str or None: Путь к выбранному файлу или None, если выбор не сделан
        """
        self.callback = callback
        
        # Этот метод должен быть переопределен в дочерних классах
        self.selected_file = None
        
        if self.callback and self.selected_file:
            self.callback(self.selected_file)
        
        return self.selected_file
    
    
    def validate_file(self, file_path: str) -> bool:
        """
        Проверяет валидность выбранного файла.
        
        Parameters:
        file_path (str): Путь к файлу для проверки
        
        Returns:
        bool: True, если файл валиден, иначе False
        """
        # Базовая проверка - просто убеждаемся, что путь не пустой
        return bool(file_path)

class FileOpenDialog(FileSelector):
    """
    Диалог для открытия файла с дополнительными возможностями настройки.
    """
    def __init__(self, parent: tk.Tk, title: str = "Открыть файл", 
                initialdir: Optional[str] = None, 
                filetypes: Optional[List[Tuple[str, str]]] = None,
                check_exists: bool = True):
        """
        Инициализация диалога открытия файла.
        
        Parameters:
        parent (tk.Tk): Родительское окно
        title (str): Заголовок диалогового окна
        initialdir (str, optional): Начальная директория
        filetypes (List[Tuple[str, str]], optional): Список поддерживаемых типов файлов
        check_exists (bool): Проверять ли существование файла
        """
        super().__init__(parent, title, initialdir, filetypes)
        self.check_exists = check_exists
        
        # Устанавливаем типы файлов по умолчанию для открытия
        if not filetypes:
            self.filetypes = [
                ("CSV файлы", "*.csv"),
                ("Excel файлы", "*.xlsx;*.xls"),
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
    
    def show(self, callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """
        Показывает диалоговое окно открытия файла.
        
        Parameters:
        callback (Callable, optional): Функция обратного вызова, вызываемая после выбора файла
        
        Returns:
        str or None: Путь к выбранному файлу или None, если выбор не сделан
        """
    
        self.callback = callback
        
        # Настраиваем цвета диалога в соответствии с темой (если поддерживается платформой)
        try:
            self.parent.tk.call('tk_getOpenFile', '-initialdir', self.initialdir,
                            '-filetypes', self.filetypes,
                            '-title', self.title,
                            '-parent', self.parent,
                            '-background', DARK_THEME['primary'],
                            '-foreground', DARK_THEME['neutral'])
        except:
            pass  # Игнорируем ошибки, если платформа не поддерживает дополнительные параметры
        
        # Открываем стандартный диалог
        self.selected_file = filedialog.askopenfilename(
            title=self.title,
            initialdir=self.initialdir,
            filetypes=self.filetypes,
            parent=self.parent
        )
        
        if self.selected_file:
            if self.validate_file(self.selected_file):
                if self.callback:
                    self.callback(self.selected_file)
                return self.selected_file
            else:
                messagebox.showerror("Ошибка", f"Выбранный файл не существует или недоступен:\n{self.selected_file}")
                return None
        
        return None
    
    def validate_file(self, file_path: str) -> bool:
        """
        Проверяет валидность выбранного файла для открытия.
        
        Parameters:
        file_path (str): Путь к файлу для проверки
        
        Returns:
        bool: True, если файл валиден, иначе False
        """
        if not super().validate_file(file_path):
            return False
        
        # Проверяем существование файла, если нужно
        if self.check_exists and not os.path.isfile(file_path):
            return False
        
        return True

class FileSaveDialog(FileSelector):
    """
    Диалог для сохранения файла с дополнительными возможностями настройки.
    """
    def __init__(self, parent: tk.Tk, title: str = "Сохранить файл", 
                initialdir: Optional[str] = None, 
                filetypes: Optional[List[Tuple[str, str]]] = None,
                defaultextension: str = ".csv",
                confirm_overwrite: bool = True):
        """
        Инициализация диалога сохранения файла.
        
        Parameters:
        parent (tk.Tk): Родительское окно
        title (str): Заголовок диалогового окна
        initialdir (str, optional): Начальная директория
        filetypes (List[Tuple[str, str]], optional): Список поддерживаемых типов файлов
        defaultextension (str): Расширение по умолчанию
        confirm_overwrite (bool): Запрашивать подтверждение при перезаписи
        """
        super().__init__(parent, title, initialdir, filetypes)
        self.defaultextension = defaultextension
        self.confirm_overwrite = confirm_overwrite
        
        # Устанавливаем типы файлов по умолчанию для сохранения
        if not filetypes:
            self.filetypes = [
                ("CSV файлы", "*.csv"),
                ("Excel файлы", "*.xlsx"),
                ("PDF файлы", "*.pdf"),
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
    
    def show(self, callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """
        Показывает диалоговое окно сохранения файла.
        
        Parameters:
        callback (Callable, optional): Функция обратного вызова, вызываемая после выбора файла
        
        Returns:
        str or None: Путь к выбранному файлу или None, если выбор не сделан
        """
        
        self.callback = callback
        
        # Настраиваем цвета диалога в соответствии с темой (если поддерживается платформой)
        try:
            self.parent.tk.call('tk_getSaveFile', '-initialdir', self.initialdir,
                            '-filetypes', self.filetypes,
                            '-title', self.title,
                            '-defaultextension', self.defaultextension,
                            '-parent', self.parent,
                            '-background', DARK_THEME['primary'],
                            '-foreground', DARK_THEME['neutral'])
        except:
            pass  # Игнорируем ошибки, если платформа не поддерживает дополнительные параметры
        
        # Открываем стандартный диалог
        self.selected_file = filedialog.asksaveasfilename(
            title=self.title,
            initialdir=self.initialdir,
            filetypes=self.filetypes,
            defaultextension=self.defaultextension,
            parent=self.parent
        )
        
        if self.selected_file:
            # Проверяем, нужно ли подтверждение перезаписи
            if os.path.exists(self.selected_file) and self.confirm_overwrite:
                # Применяем стиль темы к диалогу подтверждения
                messagebox._show(
                    "Подтверждение",
                    f"Файл {os.path.basename(self.selected_file)} уже существует.\nХотите перезаписать его?",
                    messagebox.YESNO,
                    messagebox.QUESTION,
                    parent=self.parent
                )
                if not messagebox.askyesno("Подтверждение", 
                                        f"Файл {os.path.basename(self.selected_file)} уже существует.\nХотите перезаписать его?",
                                        parent=self.parent):
                    return None
            
            if self.validate_file(self.selected_file):
                if self.callback:
                    self.callback(self.selected_file)
                return self.selected_file
        
        return None
    
    def validate_file(self, file_path: str) -> bool:
        """
        Проверяет валидность пути файла для сохранения.
        
        Parameters:
        file_path (str): Путь к файлу для проверки
        
        Returns:
        bool: True, если путь файла валиден для сохранения, иначе False
        """
        if not super().validate_file(file_path):
            return False
        
        # Проверяем, доступна ли директория для записи
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.access(directory, os.W_OK):
                messagebox.showerror("Ошибка", f"Нет прав на запись в директорию:\n{directory}")
                return False
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при проверке пути сохранения:\n{str(e)}")
            return False
            
        return True


# Вспомогательные функции для быстрого вызова диалогов

def open_file(parent: tk.Tk, title: str = "Открыть файл", 
             initialdir: Optional[str] = None,
             filetypes: Optional[List[Tuple[str, str]]] = None,
             callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Открывает диалоговое окно выбора файла и возвращает путь к выбранному файлу.
    
    Parameters:
    parent (tk.Tk): Родительское окно
    title (str): Заголовок диалогового окна
    initialdir (str, optional): Начальная директория
    filetypes (List[Tuple[str, str]], optional): Список поддерживаемых типов файлов
    callback (Callable, optional): Функция обратного вызова после выбора файла
    
    Returns:
    str or None: Путь к выбранному файлу или None, если выбор не сделан
    """
    # Применяем тему к родительскому окну
    apply_theme(parent)
    
    dialog = FileOpenDialog(parent, title, initialdir, filetypes)
    return dialog.show(callback)

def save_file(parent: tk.Tk, title: str = "Сохранить файл", 
             initialdir: Optional[str] = None,
             filetypes: Optional[List[Tuple[str, str]]] = None,
             defaultextension: str = ".csv",
             callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Открывает диалоговое окно сохранения файла и возвращает путь для сохранения.
    
    Parameters:
    parent (tk.Tk): Родительское окно
    title (str): Заголовок диалогового окна
    initialdir (str, optional): Начальная директория
    filetypes (List[Tuple[str, str]], optional): Список поддерживаемых типов файлов
    defaultextension (str): Расширение по умолчанию
    callback (Callable, optional): Функция обратного вызова после выбора файла
    
    Returns:
    str or None: Путь к файлу для сохранения или None, если выбор не сделан
    """
    # Применяем тему к родительскому окну
    apply_theme(parent)
    
    dialog = FileSaveDialog(parent, title, initialdir, filetypes, defaultextension)
    return dialog.show(callback)