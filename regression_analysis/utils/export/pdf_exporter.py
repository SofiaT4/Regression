"""
Модуль для экспорта данных и результатов анализа в формате PDF.

Содержит функции для создания PDF-документов с отчетами,
графиками и таблицами результатов регрессионного анализа.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, List, Tuple, Dict, Optional, Any, Callable
from matplotlib.figure import Figure

# Импорт из библиотеки для работы с PDF
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, ListFlowable, ListItem
    from reportlab.lib.units import inch, cm
    from matplotlib.backends.backend_pdf import PdfPages
    # Импорт для поддержки кириллицы
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Регистрируем шрифты для кириллицы
    try:
        # Пытаемся зарегистрировать DejaVu Sans - шрифт с хорошей поддержкой кириллицы
        dejavu_paths = [
            '/usr/share/fonts/TTF/DejaVuSans.ttf',  # Linux
            '/Library/Fonts/DejaVuSans.ttf',        # macOS
            'C:\\Windows\\Fonts\\DejaVuSans.ttf',   # Windows
            # Системные шрифты macOS с поддержкой кириллицы
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
            '/Library/Fonts/Arial Unicode.ttf',
            # Системные шрифты Windows с поддержкой кириллицы
            'C:\\Windows\\Fonts\\times.ttf',        # Times в Windows
            'C:\\Windows\\Fonts\\timesbd.ttf',      # Times Bold в Windows
            'C:\\Windows\\Fonts\\timesi.ttf',       # Times Italic в Windows
            'C:\\Windows\\Fonts\\timesbi.ttf',      # Times Bold Italic в Windows
            # Шрифты на случай, если остальных нет
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/Times.ttc'
        ]
        
        font_registered = False
        for font_path in dejavu_paths:
            if os.path.exists(font_path):
                try:
                    # Определяем имя шрифта в зависимости от файла
                    if 'DejaVu' in font_path:
                        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
                        font_registered = True
                        break
                    elif 'Arial' in font_path:
                        pdfmetrics.registerFont(TTFont('ArialUnicode', font_path))
                        font_registered = True
                        break
                    elif 'Helvetica' in font_path:
                        pdfmetrics.registerFont(TTFont('HelveticaUnicode', font_path))
                        font_registered = True
                        break
                    elif 'Times' in font_path or 'times' in font_path:
                        # Регистрируем разные начертания Times для Windows
                        if 'timesbd.ttf' in font_path:
                            pdfmetrics.registerFont(TTFont('TimesBold', font_path))
                        elif 'timesi.ttf' in font_path:
                            pdfmetrics.registerFont(TTFont('TimesItalic', font_path))
                        elif 'timesbi.ttf' in font_path:
                            pdfmetrics.registerFont(TTFont('TimesBoldItalic', font_path))
                        else:
                            pdfmetrics.registerFont(TTFont('TimesNewRoman', font_path))
                        font_registered = True
                        break
                except Exception as e:
                    print(f"Не удалось зарегистрировать шрифт {font_path}: {e}")
        
        # Если ни один шрифт не зарегистрирован, пробуем стандартные шрифты
        if not font_registered:
            # Используем стандартные шрифты с поддержкой кириллицы
            # Helvetica и Times - встроенные, но есть проблемы с кириллицей
            # Courier - встроенный, может поддерживать базовую кириллицу
            print("Используем стандартные шрифты PDF. Кириллица может отображаться некорректно.")
    
    except Exception as e:
        print(f"Ошибка при регистрации шрифтов: {e}")
    
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Предупреждение: библиотека ReportLab не установлена. Функции экспорта PDF будут ограничены.")

# Определяем, какой шрифт использовать
def get_cyrillic_font():
    """Возвращает имя шрифта с поддержкой кириллицы."""
    # Проверяем, какие шрифты зарегистрированы
    registered_fonts = pdfmetrics.getRegisteredFontNames() if 'pdfmetrics' in globals() else []
    
    if 'DejaVuSans' in registered_fonts:
        return 'DejaVuSans'
    elif 'TimesNewRoman' in registered_fonts:
        return 'TimesNewRoman'  # Предпочитаем Times New Roman для лучшей совместимости с Windows
    elif 'ArialUnicode' in registered_fonts:
        return 'ArialUnicode'
    elif 'HelveticaUnicode' in registered_fonts:
        return 'HelveticaUnicode'
    elif 'TimesUnicode' in registered_fonts:
        return 'TimesUnicode'
    else:
        # Возвращаем стандартный шрифт, если не нашли кириллических
        return 'Helvetica'

def export_figure_to_pdf(
    figure: Figure,
    filename: str,
    title: Optional[str] = None,
    dpi: int = 300,
    tight_layout: bool = True,
    open_after_save: bool = False
) -> bool:
    """
    Экспортирует фигуру matplotlib в PDF-файл.
    
    Parameters:
    figure (Figure): Объект фигуры matplotlib
    filename (str): Имя файла для сохранения (без расширения или с .pdf)
    title (str, optional): Заголовок в метаданных PDF
    dpi (int): Разрешение изображения
    tight_layout (bool): Использовать tight_layout для оптимизации размещения
    open_after_save (bool): Открыть PDF после сохранения
    
    Returns:
    bool: True, если экспорт успешен, иначе False
    """
    try:
        # Добавляем расширение .pdf, если его нет
        if not filename.lower().endswith('.pdf'):
            filename = filename + '.pdf'
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Применяем tight_layout, если указано
        if tight_layout:
            figure.tight_layout()
        
        # Сохраняем фигуру в PDF
        if title:
            figure.savefig(filename, dpi=dpi, bbox_inches='tight', metadata={'Title': title})
        else:
            figure.savefig(filename, dpi=dpi, bbox_inches='tight')
        
        # Открываем файл, если указано
        if open_after_save:
            try:                
                if sys.platform == 'win32':
                    os.startfile(filename)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', filename])
                else:  # linux
                    subprocess.call(['xdg-open', filename])
            except Exception as e:
                print(f"Не удалось открыть PDF-файл: {e}")
        
        return True
    
    except Exception as e:
        print(f"Ошибка при экспорте фигуры в PDF: {e}")
        return False

def export_multiple_figures(
    figures: List[Figure],
    filename: str,
    titles: Optional[List[str]] = None,
    dpi: int = 300,
    open_after_save: bool = False
) -> bool:
    """
    Экспортирует несколько фигур matplotlib в один PDF-файл.
    
    Parameters:
    figures (List[Figure]): Список объектов фигур matplotlib
    filename (str): Имя файла для сохранения (без расширения или с .pdf)
    titles (List[str], optional): Список заголовков для каждой фигуры
    dpi (int): Разрешение изображения
    open_after_save (bool): Открыть PDF после сохранения
    
    Returns:
    bool: True, если экспорт успешен, иначе False
    """
    try:
        # Добавляем расширение .pdf, если его нет
        if not filename.lower().endswith('.pdf'):
            filename = filename + '.pdf'
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Применяем tight_layout к каждой фигуре
        for fig in figures:
            fig.tight_layout()
        
        # Создаем многостраничный PDF
        with PdfPages(filename) as pdf:
            # Метаданные для PDF
            d = pdf.infodict()
            d['Title'] = 'Результаты регрессионного анализа'
            d['Author'] = 'Приложение регрессионного анализа'
            d['Subject'] = 'Статистика и графики регрессионного анализа'
            d['Keywords'] = 'регрессия, статистика, анализ данных'
            d['CreationDate'] = datetime.now()
            
            # Сохраняем каждую фигуру на отдельной странице
            for i, fig in enumerate(figures):
                # Добавляем заголовок, если он предоставлен
                if titles and i < len(titles):
                    # Создаем надстройку для заголовка, если он предоставлен
                    fig.suptitle(titles[i], fontsize=14, fontweight='bold')
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Оставляем место для заголовка
                
                pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
        
        # Открываем файл, если указано
        if open_after_save:
            try:
                # Используем глобально импортированные модули
                if sys.platform == 'win32':
                    os.startfile(filename)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', filename])
                else:  # linux
                    subprocess.call(['xdg-open', filename])
            except Exception as e:
                print(f"Не удалось открыть PDF-файл: {e}")
        
        return True
    
    except Exception as e:
        print(f"Ошибка при экспорте фигур в PDF: {e}")
        return False

def create_pdf_report(
    filename: str,
    title: str,
    content: Dict[str, Any],
    figures: Optional[List[Figure]] = None,
    tables: Optional[List[pd.DataFrame]] = None,
    table_titles: Optional[List[str]] = None,
    logo_path: Optional[str] = None,
    author: str = "Приложение регрессионного анализа",
    pagesize: str = "A4",
    open_after_save: bool = False
) -> bool:
    """
    Создает полный PDF-отчет с данными регрессионного анализа.
    
    Parameters:
    filename (str): Имя файла для сохранения (без расширения или с .pdf)
    title (str): Заголовок отчета
    content (Dict[str, Any]): Словарь с содержимым отчета (заголовки и тексты разделов)
    figures (List[Figure], optional): Список объектов фигур matplotlib
    tables (List[pd.DataFrame], optional): Список таблиц данных
    table_titles (List[str], optional): Список заголовков для таблиц
    logo_path (str, optional): Путь к логотипу
    author (str): Автор отчета
    pagesize (str): Размер страницы ("A4" или "letter")
    open_after_save (bool): Открыть PDF после сохранения
    
    Returns:
    bool: True, если создание отчета успешно, иначе False
    """
    if not REPORTLAB_AVAILABLE:
        print("Ошибка: библиотека ReportLab не установлена. Невозможно создать PDF-отчет.")
        # Пробуем сохранить хотя бы графики, если они есть
        if figures:
            return export_multiple_figures(figures, filename, open_after_save=open_after_save)
        return False
    
    try:
        # Добавляем расширение .pdf, если его нет
        if not filename.lower().endswith('.pdf'):
            filename = filename + '.pdf'
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Определяем размер страницы
        page_size = A4 if pagesize.upper() == "A4" else letter
        
        # Получаем шрифт с поддержкой кириллицы
        cyrillic_font = get_cyrillic_font()
        
        # Создаем PDF-документ
        doc = SimpleDocTemplate(
            filename,
            pagesize=page_size,
            title=title,
            author=author,
            subject="Отчет регрессионного анализа",
            keywords="регрессия, статистика, анализ данных"
        )
        
        # Получаем стили и модифицируем их для кириллицы
        styles = getSampleStyleSheet()
        
        # Создаем кастомные стили для русского текста
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=cyrillic_font,
            fontSize=18,
            alignment=1  # По центру
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontName=cyrillic_font,
            fontSize=14
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontName=cyrillic_font,
            fontSize=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=cyrillic_font,
            fontSize=10
        )
        
        # Создаем стиль для заголовков таблиц
        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=normal_style,
            fontName=cyrillic_font,
            fontSize=10,
            textColor=colors.black,
            alignment=1  # По центру
        )
        
        # Создаем элементы документа
        elements = []
        
        # Добавляем логотип, если он указан
        if logo_path and os.path.exists(logo_path):
            logo = Image(logo_path, width=1.5*inch, height=1.5*inch)
            elements.append(logo)
            elements.append(Spacer(1, 0.5*inch))
        
        # Добавляем заголовок
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Добавляем дату
        date_string = f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        elements.append(Paragraph(date_string, ParagraphStyle('Italic', parent=normal_style, fontName=cyrillic_font)))
        elements.append(Spacer(1, 0.5*inch))
        
        # Добавляем содержимое из словаря
        for section_title, section_content in content.items():
            # Заголовок раздела
            elements.append(Paragraph(section_title, heading1_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Содержимое раздела
            if isinstance(section_content, str):
                # Простой текст
                elements.append(Paragraph(section_content, normal_style))
            elif isinstance(section_content, list):
                # Список
                items = []
                for item in section_content:
                    items.append(ListItem(Paragraph(item, normal_style)))
                elements.append(ListFlowable(items, bulletType='bullet'))
            elif isinstance(section_content, dict):
                # Подразделы
                for subsection_title, subsection_content in section_content.items():
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(Paragraph(subsection_title, heading2_style))
                    elements.append(Spacer(1, 0.05*inch))
                    
                    if isinstance(subsection_content, str):
                        elements.append(Paragraph(subsection_content, normal_style))
                    elif isinstance(subsection_content, list):
                        items = []
                        for item in subsection_content:
                            items.append(ListItem(Paragraph(item, normal_style)))
                        elements.append(ListFlowable(items, bulletType='bullet'))
            
            elements.append(Spacer(1, 0.2*inch))
        
        # Добавляем таблицы данных, если они есть
        if tables:
            for i, table_df in enumerate(tables):
                # Добавляем разрыв страницы перед каждой таблицей
                elements.append(PageBreak())
                
                # Заголовок таблицы
                if table_titles and i < len(table_titles):
                    elements.append(Paragraph(table_titles[i], heading1_style))
                else:
                    elements.append(Paragraph(f"Таблица {i+1}", heading1_style))
                
                elements.append(Spacer(1, 0.1*inch))
                
                # Конвертируем DataFrame в список для ReportLab Table
                # Используем строковое представление с корректной кодировкой
                table_data = [[Paragraph(str(col), table_header_style) for col in table_df.columns]]
                
                # Преобразование всех данных в строки с корректной кодировкой
                for _, row in table_df.iterrows():
                    row_data = []
                    for val in row:
                        # Преобразуем значение в строку и оборачиваем в Paragraph для поддержки кириллицы
                        if isinstance(val, float):
                            # Форматируем числа с плавающей точкой
                            if abs(val) < 0.0001 or abs(val) > 10000:
                                text_val = f"{val:.4e}"  # Научная нотация для очень маленьких/больших чисел
                            else:
                                text_val = f"{val:.4f}"  # Обычный формат для других чисел
                        else:
                            text_val = str(val)
                        row_data.append(Paragraph(text_val, normal_style))
                    table_data.append(row_data)
                
                # Создаем таблицу
                col_widths = [doc.width/len(table_df.columns)] * len(table_df.columns)
                table = Table(table_data, colWidths=col_widths)
                
                # Стиль таблицы
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), cyrillic_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 1), (-1, -1), cyrillic_font),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ])
                
                # Чередуем цвета строк для улучшения читаемости
                for i in range(1, len(table_data), 2):
                    table_style.add('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)
                
                table.setStyle(table_style)
                elements.append(table)
                elements.append(Spacer(1, 0.2*inch))
        
        # Сохраняем временные файлы для изображений, если графики предоставлены
        temp_image_files = []
        if figures:
            for i, fig in enumerate(figures):
                # Создаем временный файл для каждой фигуры
                temp_file = f"temp_fig_{i}.png"
                fig.savefig(temp_file, dpi=150, bbox_inches='tight')
                temp_image_files.append(temp_file)
                
                # Добавляем разрыв страницы и изображение
                elements.append(PageBreak())
                
                # Заголовок для графика
                if figures and i < len(figures):
                    if hasattr(fig, 'suptitle') and fig._suptitle:
                        chart_title = fig._suptitle.get_text()
                    else:
                        axes = fig.get_axes()
                        if axes and hasattr(axes[0], 'get_title'):
                            chart_title = axes[0].get_title()
                        else:
                            chart_title = f"График {i+1}"
                    
                    elements.append(Paragraph(chart_title, heading1_style))
                    elements.append(Spacer(1, 0.1*inch))
                
                # Добавляем изображение
                img = Image(temp_file, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        
        # Создаем PDF-документ
        doc.build(elements)
        
        # Удаляем временные файлы изображений
        for temp_file in temp_image_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Открываем файл, если указано
        if open_after_save:
            try:
                # Используем глобально импортированные модули
                if sys.platform == 'win32':
                    os.startfile(filename)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', filename])
                else:  # linux
                    subprocess.call(['xdg-open', filename])
            except Exception as e:
                print(f"Не удалось открыть PDF-файл: {e}")
        
        return True
    
    except Exception as e:
        print(f"Ошибка при создании PDF-отчета: {e}")
        return False

def export_to_pdf(
    data: Union[Dict[str, Any], pd.DataFrame],
    filename: str,
    title: str = "Результаты регрессионного анализа",
    include_figures: Optional[List[Figure]] = None,
    report_type: str = "simple",
    open_after_save: bool = False
) -> bool:
    """
    Универсальная функция для экспорта данных в PDF.
    
    Parameters:
    data (Dict or DataFrame): Данные для экспорта
    filename (str): Имя файла для сохранения
    title (str): Заголовок документа
    include_figures (List[Figure], optional): Графики для включения в отчет
    report_type (str): Тип отчета ("simple", "detailed", "coefficients", "statistics")
    open_after_save (bool): Открыть PDF после сохранения
    
    Returns:
    bool: True, если экспорт успешен, иначе False
    """
    # Если переданы только графики, используем специализированную функцию
    if data is None and include_figures:
        return export_multiple_figures(include_figures, filename, open_after_save=open_after_save)
    
    # Определяем тип данных и выбираем стратегию экспорта
    if isinstance(data, pd.DataFrame):
        # Для DataFrame создаем простой отчет с таблицей
        tables = [data]
        table_titles = ["Данные"]
        
        content = {
            "Сводная информация": f"Анализ датасета с {data.shape[0]} наблюдениями и {data.shape[1]} признаками.",
            "Структура данных": {
                "Количество строк": str(data.shape[0]),
                "Количество столбцов": str(data.shape[1]),
                "Типы данных": [f"{col}: {str(dtype)}" for col, dtype in data.dtypes.items()]
            }
        }
        
        return create_pdf_report(
            filename=filename,
            title=title,
            content=content,
            figures=include_figures,
            tables=tables,
            table_titles=table_titles,
            open_after_save=open_after_save
        )
    
    elif isinstance(data, dict):
        # Для словаря выбираем стратегию в зависимости от типа отчета
        if report_type == "coefficients":
            # Отчет с коэффициентами модели
            if "coefficients" in data and "feature_names" in data:
                # Создаем DataFrame из коэффициентов
                coef_df = pd.DataFrame({
                    "Признак": data["feature_names"],
                    "Коэффициент": data["coefficients"],
                    "Стандартная ошибка": data.get("se_coefficients", [None] * len(data["coefficients"])),
                    "t-статистика": data.get("t_values", [None] * len(data["coefficients"])),
                    "P-значение": data.get("p_values", [None] * len(data["coefficients"]))
                })
                
                # Если есть доверительные интервалы, добавляем их
                if "lower_ci" in data and "upper_ci" in data:
                    coef_df["Нижний CI 95%"] = data["lower_ci"]
                    coef_df["Верхний CI 95%"] = data["upper_ci"]
                
                tables = [coef_df]
                table_titles = ["Коэффициенты регрессии"]
                
                content = {
                    "Сводная информация": "Результаты оценки коэффициентов регрессионной модели.",
                    "Показатели качества модели": {
                        "R-квадрат": f"{data.get('r2', 'Н/Д'):.4f}",
                        "Скорректированный R-квадрат": f"{data.get('adjusted_r2', 'Н/Д'):.4f}",
                        "F-статистика": f"{data.get('f_statistic', 'Н/Д'):.4f}",
                        "Значимость F": f"{data.get('p_value_f', 'Н/Д'):.4f}"
                    }
                }
                
                return create_pdf_report(
                    filename=filename,
                    title=title,
                    content=content,
                    figures=include_figures,
                    tables=tables,
                    table_titles=table_titles,
                    open_after_save=open_after_save
                )
                
        elif report_type == "statistics":
            # Отчет со статистикой модели
            content = {
                "Регрессионная статистика": {
                    "Множественный R": f"{data.get('multiple_r', 'Н/Д'):.4f}",
                    "R-квадрат": f"{data.get('r2', 'Н/Д'):.4f}",
                    "Скорректированный R-квадрат": f"{data.get('adjusted_r2', 'Н/Д'):.4f}",
                    "Стандартная ошибка": f"{data.get('se_regression', 'Н/Д'):.4f}",
                    "Наблюдений": str(data.get('observations', 'Н/Д'))
                }
            }
            
            # Если есть коэффициенты, добавляем их таблицу
            if "coefficients" in data and "feature_names" in data:
                coef_df = pd.DataFrame({
                    "Признак": data["feature_names"],
                    "Коэффициент": data["coefficients"],
                    "Стандартная ошибка": data.get("se_coefficients", [None] * len(data["coefficients"])),
                    "t-статистика": data.get("t_values", [None] * len(data["coefficients"])),
                    "P-значение": data.get("p_values", [None] * len(data["coefficients"]))
                })
                
                tables = [coef_df]
                table_titles = ["Коэффициенты регрессии"]
                
                return create_pdf_report(
                    filename=filename,
                    title=title,
                    content=content,
                    figures=include_figures,
                    tables=tables,
                    table_titles=table_titles,
                    open_after_save=open_after_save
                )
            else:
                return create_pdf_report(
                    filename=filename,
                    title=title,
                    content=content,
                    figures=include_figures,
                    open_after_save=open_after_save
                )
                
        elif report_type == "detailed":
            # Подробный отчет с максимумом информации
            # Формируем содержимое на основе доступных данных
            content = {"Введение": "Полный отчет о результатах регрессионного анализа."}
            
            # Регрессионная статистика
            if any(key in data for key in ['r2', 'adjusted_r2', 'multiple_r']):
                content["Регрессионная статистика"] = {
                    "Множественный R": f"{data.get('multiple_r', 'Н/Д'):.4f}",
                    "R-квадрат": f"{data.get('r2', 'Н/Д'):.4f}",
                    "Скорректированный R-квадрат": f"{data.get('adjusted_r2', 'Н/Д'):.4f}",
                    "Стандартная ошибка": f"{data.get('se_regression', 'Н/Д'):.4f}",
                    "Наблюдений": str(data.get('observations', 'Н/Д'))
                }
            
            # Интерпретация результатов
            if 'r2' in data and 'p_value_f' in data:
                # Добавляем раздел с интерпретацией
                interpret_text = []
                
                # Качество модели
                r2 = data.get('r2', 0)
                if r2 > 0.9:
                    interpret_text.append("Модель имеет очень высокое качество прогноза (R² > 0.9).")
                elif r2 > 0.7:
                    interpret_text.append("Модель имеет высокое качество прогноза (0.7 < R² < 0.9).")
                elif r2 > 0.5:
                    interpret_text.append("Модель имеет среднее качество прогноза (0.5 < R² < 0.7).")
                elif r2 > 0.3:
                    interpret_text.append("Модель имеет низкое качество прогноза (0.3 < R² < 0.5).")
                else:
                    interpret_text.append("Модель имеет очень низкое качество прогноза (R² < 0.3).")
                
                # Оценка значимости модели
                p_value_f = data.get('p_value_f', 1)
                if p_value_f < 0.001:
                    interpret_text.append("Модель высоко статистически значима (p < 0.001).")
                elif p_value_f < 0.01:
                    interpret_text.append("Модель статистически значима (p < 0.01).")
                elif p_value_f < 0.05:
                    interpret_text.append("Модель статистически значима (p < 0.05).")
                else:
                    interpret_text.append("Модель статистически незначима (p ≥ 0.05).")
                
                # Анализ коэффициентов
                if "coefficients" in data and "p_values" in data and "feature_names" in data:
                    # Находим значимые коэффициенты
                    significant_coefs = []
                    for i, (name, coef, p) in enumerate(zip(data["feature_names"], data["coefficients"], data["p_values"])):
                        if p < 0.05 and i > 0:  # Пропускаем константу (i=0)
                            sign = "положительное" if coef > 0 else "отрицательное"
                            significant_coefs.append(f"{name}: {sign} влияние (β={coef:.4f}, p={p:.4f})")
                    
                    if significant_coefs:
                        interpret_text.append("Значимые предикторы:")
                        interpret_text.extend(significant_coefs)
                    else:
                        interpret_text.append("Не найдено статистически значимых предикторов.")
                
                content["Интерпретация результатов"] = interpret_text
            
            # Создаем таблицы
            tables = []
            table_titles = []
            
            # Таблица коэффициентов
            if "coefficients" in data and "feature_names" in data:
                coef_df = pd.DataFrame({
                    "Признак": data["feature_names"],
                    "Коэффициент": data["coefficients"],
                    "Стандартная ошибка": data.get("se_coefficients", [None] * len(data["coefficients"])),
                    "t-статистика": data.get("t_values", [None] * len(data["coefficients"])),
                    "P-значение": data.get("p_values", [None] * len(data["coefficients"]))
                })
                
                # Если есть доверительные интервалы, добавляем их
                if "lower_ci" in data and "upper_ci" in data:
                    coef_df["Нижний CI 95%"] = data["lower_ci"]
                    coef_df["Верхний CI 95%"] = data["upper_ci"]
                
                tables.append(coef_df)
                table_titles.append("Коэффициенты регрессии")
            
            return create_pdf_report(
                filename=filename,
                title=title,
                content=content,
                figures=include_figures,
                tables=tables,
                table_titles=table_titles,
                open_after_save=open_after_save
            )
        
        else:  # report_type == "simple"
            # Простой отчет с основной статистикой
            content = {
                "Регрессионная статистика": f"R-квадрат: {data.get('r2', 'Н/Д'):.4f}\n"
                                        f"Скорректированный R-квадрат: {data.get('adjusted_r2', 'Н/Д'):.4f}\n"
                                        f"Наблюдений: {data.get('observations', 'Н/Д')}"
            }
            
            return create_pdf_report(
                filename=filename,
                title=title,
                content=content,
                figures=include_figures,
                open_after_save=open_after_save
            )
    
    else:
        # Для других типов данных пытаемся сделать простой отчет
        content = {
            "Предупреждение": "Данные неизвестного формата. Отчет может содержать неполную информацию."
        }
        
        return create_pdf_report(
            filename=filename,
            title=title,
            content=content,
            figures=include_figures,
            open_after_save=open_after_save
        )