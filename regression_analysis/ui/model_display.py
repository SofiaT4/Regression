import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

from ui.viewers.graph_viewer import GraphViewer
from ui.viewers.coefficient_viewer import CoefficientViewer
from utils.visualization.graph_manager import export_all_plots
from utils.export.pdf_exporter import export_to_pdf
from core.models.model_formatter import format_equation_for_display, format_equation_for_charts
from ui.components.theme_manager import DARK_THEME, style_treeview_tags

class ModelDisplayFrame(tk.Frame):
    """
    Фрейм для отображения результатов регрессионного анализа,
    включая статистику моделей и графики.
    """
    def __init__(self, parent, df, stats_dict, models, predictions, 
             X_all_groups, X_unemployed, X_combined, y, 
             age_groups, back_callback):
        """
        Инициализация фрейма отображения моделей.
        
        Parameters:
        parent (tk.Tk or tk.Frame): Родительский виджет
        df (pandas.DataFrame): Исходные данные
        stats_dict (dict): Словарь со статистикой моделей
        models (dict): Словарь с обученными моделями регрессии
        predictions (dict): Словарь с предсказаниями моделей
        X_all_groups, X_unemployed, X_combined: Наборы признаков для разных моделей
        y (pandas.Series): Целевая переменная
        age_groups (list): Список возрастных групп
        back_callback (function): Функция для возврата к начальному экрану
        """
        # Импортируем компоненты темы
        from ui.components.theme_manager import DARK_THEME, style_treeview_tags
        
        # Сначала создаем родительский фрейм содержащий полосу прокрутки
        self.parent = parent
        self.outer_frame = tk.Frame(parent, bg=DARK_THEME['primary'])
        self.outer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Canvas и Scrollbar
        self.canvas = tk.Canvas(
            self.outer_frame, 
            borderwidth=0, 
            highlightthickness=0,
            bg=DARK_THEME['primary']
        )
        self.vsb = tk.Scrollbar(
            self.outer_frame, 
            orient=tk.VERTICAL, 
            command=self.canvas.yview,
            bg=DARK_THEME['bg_light'],
            troughcolor=DARK_THEME['primary']
        )
        self.canvas.configure(yscrollcommand=self.vsb.set)
        
        # Размещаем Canvas и Scrollbar
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Создаем фрейм внутри Canvas, который будет содержать все
        super().__init__(self.canvas, bg=DARK_THEME['primary'])
        self.canvas_frame = self.canvas.create_window((0, 0), window=self, anchor="nw")
        
        # Настраиваем прокрутку мыши
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Configure>", self._on_frame_configure)
        
        # Инициализируем данные
        self.df = df
        self.stats_dict = stats_dict
        self.models = models
        self.predictions = predictions
        self.X_all_groups = X_all_groups
        self.X_unemployed = X_unemployed
        self.X_combined = X_combined
        self.y = y
        self.age_groups = age_groups
        self.back_callback = back_callback
        self.graph_window = None
        
        # По умолчанию отображаем комбинированную модель, если доступна
        self.current_model = 'combined'
        if not age_groups:  # Если возрастные группы не найдены, показываем модель безработицы
            self.current_model = 'unemployed'
        
        # Делаем главное окно растягиваемым
        self.parent.resizable(True, True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса с вкладками."""
        from ui.components.theme_manager import DARK_THEME, style_treeview_tags
        
        # Заголовок
        title_label = tk.Label(
            self, 
            text="Результаты регрессионного анализа", 
            font=("Arial", 16, "bold"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        title_label.pack(pady=10)
        
        # Создаем стиль для вкладок
        style = ttk.Style()
        style.configure(
            "TNotebook", 
            background=DARK_THEME['primary'],
            borderwidth=0
        )
        style.configure(
            "TNotebook.Tab", 
            background=DARK_THEME['bg_light'],
            foreground=DARK_THEME['neutral'],
            padding=[10, 2],
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[('selected', DARK_THEME['accent']), ('active', DARK_THEME['bg'])],
            foreground=[('selected', DARK_THEME['text_light']), ('active', DARK_THEME['text_light'])]
        )
        
        # Создаем вкладки для отображения результатов
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Вкладка "Регрессионная статистика"
        self.stats_frame = tk.Frame(self.notebook, padx=10, pady=10, bg=DARK_THEME['primary'])
        self.notebook.add(self.stats_frame, text="Регрессионная статистика")
        
        # Вкладка "Графики"
        self.graphs_frame = tk.Frame(self.notebook, padx=10, pady=10, bg=DARK_THEME['primary'])
        self.notebook.add(self.graphs_frame, text="Графики")
        
        # Настройка вкладок
        self.setup_statistics_tab()
        self.setup_graphs_tab()
        
        # Кнопки управления
        button_frame = tk.Frame(self, bg=DARK_THEME['primary'])
        button_frame.pack(fill=tk.X, pady=10)
        
        # Кнопка экспорта отчета в PDF
        report_button = tk.Button(
            button_frame, 
            text="Экспортировать отчет",  
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.export_report
        )
        report_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Кнопка экспорта графиков в PDF
        save_button = tk.Button(
            button_frame, 
            text="Экспортировать графики",  
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.export_current_model_plots
        )
        save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Кнопка возврата к начальному экрану
        back_button = tk.Button(
            button_frame, 
            text="Вернуться к выбору файла", 
            font=("Arial", 12),
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.handle_back_button  # Use the new handler
        )
        back_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
    def update_scrollregion(self, event):
        """Обновляет регион прокрутки при изменении содержимого."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Обработка события прокрутки колесом мыши."""
        # Направление прокрутки для разных платформ
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def on_mousewheel_linux(self, event):
        """Обработчик прокрутки колесом мыши для Linux."""
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def toggle_section(self, section_frame):
        """Сворачивает/разворачивает секцию и обновляет область прокрутки."""
        # Инвертируем состояние
        is_expanded = not section_frame.is_expanded.get()
        section_frame.is_expanded.set(is_expanded)
        
        # Обновляем символ кнопки
        section_frame.toggle_button.config(
            text="▼" if is_expanded else "►"
        )
        
        # Скрываем/показываем содержимое
        if is_expanded:
            section_frame.contentframe.pack(fill=tk.X, padx=20, pady=5)
        else:
            section_frame.contentframe.pack_forget()
        
        # Обновляем регион прокрутки
        self.update_scrollregion(None)

    def _on_frame_configure(self, event):
        """Обновление области прокрутки при изменении размера внутреннего фрейма."""
        # Обновляем размеры прокручиваемой области
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_canvas_configure(self, event):
        """Изменение размера внутреннего фрейма при изменении размера холста."""
        # Обновляем ширину внутреннего фрейма
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def destroy(self):
        """Properly clean up the frame and all its components."""
        # Store references to widgets we want to destroy
        canvas = self.canvas
        vsb = self.vsb
        outer_frame = self.outer_frame
        
        # First unbind all events to prevent issues
        try:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        except:
            pass  # Ignore any errors from unbinding
        
        # Remove our instance from the parent to prevent circular references
        if hasattr(self, 'parent') and hasattr(self.parent, 'results_frame'):
            self.parent.results_frame = None
        
        # Call super's destroy method first
        super().destroy()
        
        # Now safely destroy components
        try:
            canvas.destroy()
        except:
            pass
            
        try:
            vsb.destroy()
        except:
            pass
            
        try:
            outer_frame.destroy()
        except:
            pass

    def bind_mousewheel(self):
        """Привязывает колесо мыши к прокрутке canvas."""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Для Windows
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Для Linux
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
    
    def setup_statistics_tab(self):
        """Настройка вкладки со статистикой регрессии."""
        from ui.components.theme_manager import DARK_THEME
        
        # Выбор модели для отображения
        model_frame = tk.LabelFrame(
            self.stats_frame, 
            text="Выбор модели", 
            padx=10, 
            pady=10,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        model_frame.pack(fill=tk.X, pady=5)
        
        # Радиокнопки для выбора модели
        self.model_var = tk.StringVar(value=self.current_model)
        
        # Активируем только те модели, которые доступны
        if self.age_groups:
            rb_all_groups = tk.Radiobutton(
                model_frame, 
                text="ВВП от Численности рабочих", 
                variable=self.model_var, value="all_groups",
                command=self.change_model,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            rb_all_groups.pack(anchor=tk.W, padx=5, pady=2)
            
            rb_combined = tk.Radiobutton(
                model_frame, 
                text="ВВП от Численности рабочих и Безработицы", 
                variable=self.model_var, value="combined",
                command=self.change_model,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            rb_combined.pack(anchor=tk.W, padx=5, pady=2)
        
        rb_unemployed = tk.Radiobutton(
            model_frame, 
            text="ВВП от Безработицы", 
            variable=self.model_var, value="unemployed",
            command=self.change_model,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral'],
            selectcolor=DARK_THEME['accent'],
            activebackground=DARK_THEME['primary'],
            activeforeground=DARK_THEME['text_light']
        )
        rb_unemployed.pack(anchor=tk.W, padx=5, pady=2)
        
        # Создаем фрейм для содержимого, которое будет меняться при выборе модели
        self.content_frame = tk.Frame(self.stats_frame, bg=DARK_THEME['primary'])
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Создаем сворачиваемые разделы
        self.equation_frame = self.create_collapsible_section(self.stats_frame, "Уравнение регрессии")
        self.reg_stats_frame = self.create_collapsible_section(self.stats_frame, "Регрессионная статистика")
        self.anova_frame = self.create_collapsible_section(self.stats_frame, "Дисперсионный анализ")
        self.coef_frame = self.create_collapsible_section(self.stats_frame, "Коэффициенты")
        
        # Добавляем сворачиваемую секцию для сравнения моделей
        self.comparison_frame = self.create_collapsible_section(self.stats_frame, "Сравнение моделей регрессии")
        self.setup_model_comparison_section(self.comparison_frame)
         
        # Заполняем контент для выбранной модели
        self.update_content()
        
    def create_collapsible_section(self, parent, title):
        """
        Создает сворачиваемую секцию с заголовком.
        
        Parameters:
        parent (tk.Widget): Родительский виджет
        title (str): Заголовок секции
        
        Returns:
        tk.Frame: Фрейм секции
        """
        from ui.components.theme_manager import DARK_THEME
        
        frame = tk.Frame(parent, bg=DARK_THEME['primary'])
        frame.pack(fill=tk.X, pady=5)
        
        title_button_frame = tk.Frame(frame, bg=DARK_THEME['primary'])
        title_button_frame.pack(fill=tk.X)
        
        # Создаем переменную для отслеживания состояния раскрытия
        frame.is_expanded = tk.BooleanVar(value=True)
        
        # Символы для состояний разворачивания/сворачивания
        expand_symbol = "▼"  # Unicode символ треугольника вниз
        collapse_symbol = "►"  # Unicode символ треугольника вправо
        
        # Создаем кнопку разворачивания/сворачивания
        frame.toggle_button = tk.Button(
            title_button_frame, 
            text=expand_symbol if frame.is_expanded.get() else collapse_symbol,
            width=2,
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'], 
            command=lambda: self.toggle_section(frame)
        )
        frame.toggle_button.pack(side=tk.LEFT)
        
        # Добавляем заголовок
        tk.Label(
            title_button_frame, 
            text=title, 
            font=("Arial", 12, "bold"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        ).pack(side=tk.LEFT, padx=5)
        
        # Создаем содержимое секции, которое будет скрываться/показываться
        frame.contentframe = tk.Frame(frame, bg=DARK_THEME['primary'])
        frame.contentframe.pack(fill=tk.X, padx=20, pady=5)
        
        return frame
    
    def toggle_section(self, section_frame):
        """Сворачивает/разворачивает секцию."""
        # Инвертируем состояние
        is_expanded = not section_frame.is_expanded.get()
        section_frame.is_expanded.set(is_expanded)
        
        # Обновляем символ кнопки
        section_frame.toggle_button.config(
            text="▼" if is_expanded else "►"
        )
        
        # Скрываем/показываем содержимое
        if is_expanded:
            section_frame.contentframe.pack(fill=tk.X, padx=20, pady=5)
        else:
            section_frame.contentframe.pack_forget()
        
        # Обновляем область прокрутки после изменения содержимого
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def setup_model_comparison_section(self, parent_frame):
        """
        Настройка секции сравнения моделей.
        """
        content_frame = parent_frame.contentframe
        
        # Очищаем все существующие виджеты в content_frame
        for widget in content_frame.winfo_children():
            widget.destroy()
        
        # Информационный заголовок
        info_label = tk.Label(
            content_frame, 
            text="Таблица показывает сравнение статистических показателей всех моделей регрессии",
            font=("Arial", 10, "italic"), 
            wraplength=700,
            justify=tk.LEFT,
            pady=5
        )
        info_label.pack(anchor=tk.W, fill=tk.X)
        
        # Создание фрейма для таблицы с явной высотой
        table_container = tk.Frame(content_frame, height=400, width=750)
        table_container.pack(fill=tk.BOTH, expand=True, pady=10)
        table_container.pack_propagate(False)  # Предотвращаем изменение размера фрейма
        
        # Создаем Treeview для таблицы
        columns = ("1", "2", "3", "4")
        self.comparison_table = ttk.Treeview(
            table_container, 
            columns=columns, 
            show="headings", 
            height=10
        )
        
        # Настраиваем заголовки и ширину столбцов
        self.comparison_table.heading("1", text="Показатель")
        self.comparison_table.heading("2", text="Модель от численности рабочих")
        self.comparison_table.heading("3", text="Модель от безработицы")
        self.comparison_table.heading("4", text="Комбинированная модель")
        
        self.comparison_table.column("1", width=250, anchor=tk.W)
        self.comparison_table.column("2", width=150, anchor=tk.CENTER)
        self.comparison_table.column("3", width=150, anchor=tk.CENTER)
        self.comparison_table.column("4", width=150, anchor=tk.CENTER)
        
        # Настраиваем теги для форматирования строк
        self.comparison_table.tag_configure("header", background="#f0f0f0", font=("Arial", 9, "bold"))
        self.comparison_table.tag_configure("subheader", background="#f8f8f8")
        self.comparison_table.tag_configure("best", background="#e6ffe6")  # Светло-зеленый для лучших показателей
        self.comparison_table.tag_configure("worst", background="#ffe6e6")  # Светло-красный для худших показателей
        self.comparison_table.tag_configure("separator", background="#f5f5f5")
        
        # Полосы прокрутки
        scrollbar_y = ttk.Scrollbar(table_container, orient="vertical", command=self.comparison_table.yview)
        scrollbar_x = ttk.Scrollbar(table_container, orient="horizontal", command=self.comparison_table.xview)
        
        # Размещение полос прокрутки
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Размещение таблицы
        self.comparison_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Связываем таблицу с полосами прокрутки
        self.comparison_table.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Добавляем кнопки
        buttons_frame = tk.Frame(content_frame, pady=10)
        buttons_frame.pack(fill=tk.X)
        
        # Кнопка экспорта
        export_button = tk.Button(
            buttons_frame,
            text="Экспортировать сравнение в CSV",
            command=self.export_comparison_to_csv,
            font=("Arial", 10)
        )
        export_button.pack(side=tk.LEFT, padx=5)
          
        # Заполняем таблицу реальными данными
        self.populate_comparison_table()
        
    def export_comparison_to_csv(self):
        """Экспортирует таблицу сравнения моделей в CSV файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            title="Экспортировать сравнение моделей в CSV"
        )
        
        if not file_path:
            return
            
        try:
            # Создаем DataFrame для экспорта
            data = []
            
            # Получаем данные из таблицы
            for item_id in self.comparison_table.get_children():
                values = self.comparison_table.item(item_id, "values")
                if values:
                    data.append(values)
            
            # Создаем DataFrame и экспортируем в CSV
            if data:
                df = pd.DataFrame(data, columns=["Показатель", "Модель от численности рабочих", 
                                            "Модель от безработицы", "Комбинированная модель"])
                df.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')
                messagebox.showinfo("Экспорт", f"Сравнение моделей экспортировано в файл:\n{file_path}")
            else:
                messagebox.showwarning("Экспорт", "Нет данных для экспорта.")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте сравнения:\n{str(e)}")

    def setup_graphs_tab(self):
        """Настройка вкладки с графиками."""
        from ui.components.theme_manager import DARK_THEME
        
        graph_titles = [
            "1. Фактический и прогнозируемый ВВП",
            "2. Визуализация коэффициентов модели",
            "3. График остатков",
            "4. Динамика показателей (нормализованные значения)"
        ]
        
        # Выбор модели для графиков
        model_frame = tk.LabelFrame(
            self.graphs_frame, 
            text="Выбор модели", 
            padx=10, 
            pady=5,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        model_frame.pack(fill=tk.X, pady=5)
        
        graph_model_var = tk.StringVar(value=self.current_model)
        
        # Функция для обновления выбранной модели для графиков
        def update_graph_model():
            self.current_model = graph_model_var.get()
        
        # Радиокнопки для моделей
        if self.age_groups:
            rb_all_groups = tk.Radiobutton(
                model_frame, 
                text="Модель от численности рабочих", 
                variable=graph_model_var, 
                value="all_groups",
                command=update_graph_model,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            rb_all_groups.pack(side=tk.LEFT, padx=5)
            
            rb_combined = tk.Radiobutton(
                model_frame, 
                text="Комбинированная модель", 
                variable=graph_model_var, 
                value="combined",
                command=update_graph_model,
                bg=DARK_THEME['primary'],
                fg=DARK_THEME['neutral'],
                selectcolor=DARK_THEME['accent'],
                activebackground=DARK_THEME['primary'],
                activeforeground=DARK_THEME['text_light']
            )
            rb_combined.pack(side=tk.LEFT, padx=5)
        
        rb_unemployed = tk.Radiobutton(
            model_frame, 
            text="Модель от безработицы", 
            variable=graph_model_var, 
            value="unemployed",
            command=update_graph_model,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral'],
            selectcolor=DARK_THEME['accent'],
            activebackground=DARK_THEME['primary'],
            activeforeground=DARK_THEME['text_light']
        )
        rb_unemployed.pack(side=tk.LEFT, padx=5)
        
        # Создаем фрейм для списка графиков
        graphs_list_frame = tk.Frame(self.graphs_frame, pady=10, bg=DARK_THEME['primary'])
        graphs_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок для списка графиков
        tk.Label(
            graphs_list_frame, 
            text="Доступные графики:", 
            font=("Arial", 12, "bold"),
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        ).pack(anchor=tk.W, pady=5)
        
        # Создаем кнопки для каждого типа графика
        for i, title in enumerate(graph_titles):
            button = tk.Button(
                graphs_list_frame, 
                text=title, 
                font=("Arial", 12),
                anchor="w",
                padx=10,
                pady=5,
                width=50,
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light'],
                command=lambda idx=i: self.show_graph(idx)
            )
            button.pack(fill=tk.X, pady=2)
            
        button = tk.Button(
            graphs_list_frame, 
            text="5. Сравнить модели графически", 
            font=("Arial", 12),
            anchor="w",
            padx=10,
            pady=5,
            width=50,
            bg=DARK_THEME['bg_light'],
            fg=DARK_THEME['neutral'],
            activebackground=DARK_THEME['accent'],
            activeforeground=DARK_THEME['text_light'],
            command=self.show_graphical_comparison
        )
        button.pack(fill=tk.X, pady=2)
        
    def change_model(self):
        """Обработчик переключения между моделями."""
        self.current_model = self.model_var.get()
        self.update_content()
    
    def update_content(self):
        """Обновляет содержимое при смене модели."""
        from ui.components.theme_manager import DARK_THEME
        
        # Очищаем текущее содержимое всех секций
        for frame in [self.equation_frame.contentframe, self.reg_stats_frame.contentframe, 
                    self.anova_frame.contentframe, self.coef_frame.contentframe]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Получаем статистику для выбранной модели
        model_stats = self.stats_dict[self.current_model]
        
        # 1. Уравнение регрессии в формате Excel
        # Форматируем уравнение регрессии в зависимости от модели
        if self.current_model == 'all_groups':
            equation_text = format_equation_for_display(model_stats, self.current_model, "Численность рабочих", [])
        elif self.current_model == 'unemployed':
            equation_text = format_equation_for_display(model_stats, self.current_model, "", ["Безработица"])
        else:  # combined
            equation_text = format_equation_for_display(model_stats, self.current_model, "Численность рабочих", ["Безработица"])
        
        equation_label = tk.Label(
            self.equation_frame.contentframe, 
            text=equation_text, 
            font=("Courier", 10), 
            wraplength=600,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        equation_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Добавляем уравнение для графиков
        chart_eq_text = format_equation_for_charts(model_stats, self.current_model)
        chart_eq_label = tk.Label(
            self.equation_frame.contentframe, 
            text=f"Для графиков: {chart_eq_text}", 
            font=("Courier", 10), 
            wraplength=600,
            bg=DARK_THEME['primary'],
            fg=DARK_THEME['neutral']
        )
        chart_eq_label.pack(anchor=tk.W, padx=5, pady=5)

        # Настройка стиля для ttk виджетов
        style = ttk.Style()
        style.configure(
            "TLabel", 
            background=DARK_THEME['primary'],
            foreground=DARK_THEME['neutral']
        )
        style.configure(
            "Bold.TLabel", 
            background=DARK_THEME['primary'],
            foreground=DARK_THEME['neutral'],
            font=("Arial", 10, "bold")
        )
        
        # 2. Заполняем регрессионную статистику
        # Настраиваем сетку для таблицы регрессионной статистики
        reg_stats_frame = self.reg_stats_frame.contentframe
        reg_stats_frame.columnconfigure(0, weight=1)
        reg_stats_frame.columnconfigure(1, weight=1)
        
        # Статистические показатели из Excel с использованием стилей
        ttk.Label(reg_stats_frame, text="Множественный R:", style="Bold.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(reg_stats_frame, text=f"{model_stats['multiple_r']:.8f}", style="TLabel").grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(reg_stats_frame, text="R-квадрат:", style="Bold.TLabel").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(reg_stats_frame, text=f"{model_stats['r2']:.8f}", style="TLabel").grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(reg_stats_frame, text="Нормированный R-квадрат:", style="Bold.TLabel").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(reg_stats_frame, text=f"{model_stats['adjusted_r2']:.8f}", style="TLabel").grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(reg_stats_frame, text="Стандартная ошибка:", style="Bold.TLabel").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(reg_stats_frame, text=f"{model_stats['se_regression']:.8f}", style="TLabel").grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(reg_stats_frame, text="Наблюдения:", style="Bold.TLabel").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(reg_stats_frame, text=f"{model_stats['observations']}", style="TLabel").grid(row=4, column=1, sticky="w", padx=5, pady=2)
        
        # 3. Заполняем дисперсионный анализ
        anova_frame = self.anova_frame.contentframe
        
        # Создаем Treeview для таблицы дисперсионного анализа
        columns = ("1", "2", "3", "4", "5")
        anova_table = ttk.Treeview(anova_frame, columns=columns, show="headings", height=3)
        
        # Настраиваем заголовки и ширину столбцов
        anova_table.heading("1", text="")
        anova_table.heading("2", text="df")
        anova_table.heading("3", text="SS")
        anova_table.heading("4", text="MS")
        anova_table.heading("5", text="F")
        
        anova_table.column("1", width=100, anchor=tk.W)
        anova_table.column("2", width=70, anchor=tk.CENTER)
        anova_table.column("3", width=150, anchor=tk.CENTER)
        anova_table.column("4", width=150, anchor=tk.CENTER)
        anova_table.column("5", width=100, anchor=tk.CENTER)
        
        # Заполняем таблицу ANOVA
        anova_table.insert("", "end", values=(
            "Регрессия", 
            f"{model_stats['df_regression']}", 
            f"{model_stats['ss_regression']:.8f}", 
            f"{model_stats['ms_regression']:.8f}", 
            f"{model_stats['f_statistic']:.8f}"
        ))
        
        anova_table.insert("", "end", values=(
            "Остаток", 
            f"{model_stats['df_residual']}", 
            f"{model_stats['ss_residual']:.8f}", 
            f"{model_stats['ms_residual']:.8f}", 
            ""
        ))
        
        anova_table.insert("", "end", values=(
            "Итого", 
            f"{model_stats['df_total']}", 
            f"{model_stats['ss_total']:.8f}", 
            "", 
            ""
        ))
        
        # Применяем стили темной темы к таблице
        style_treeview_tags(anova_table)

        anova_table.pack(fill=tk.X, padx=5, pady=5)
        
        # Добавляем информацию о значимости F под таблицей ANOVA
        f_info_frame = tk.Frame(anova_frame, bg=DARK_THEME['primary'])
        f_info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(f_info_frame, text="Значимость F:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_info_frame, text=f"{model_stats['p_value_f']:.8f}").pack(side=tk.LEFT, padx=5)
        
        # 4. Заполняем коэффициенты
        coef_frame = self.coef_frame.contentframe
        
        # Создаем Treeview для таблицы коэффициентов
        columns = ("1", "2", "3", "4", "5", "6", "7")
        coef_table = ttk.Treeview(coef_frame, columns=columns, show="headings", height=min(len(model_stats['coefficients']), 6))
        
        # Настраиваем заголовки и ширину столбцов
        coef_table.heading("1", text="")
        coef_table.heading("2", text="Коэффициенты")
        coef_table.heading("3", text="Стандартная ошибка")
        coef_table.heading("4", text="t-статистика")
        coef_table.heading("5", text="P-Значение")
        coef_table.heading("6", text="Нижние 95%")
        coef_table.heading("7", text="Верхние 95%")
        
        coef_table.column("1", width=150)
        coef_table.column("2", width=100)
        coef_table.column("3", width=120)
        coef_table.column("4", width=100)
        coef_table.column("5", width=100)
        coef_table.column("6", width=100)
        coef_table.column("7", width=100)
        
        # Используем Excel-форматированные имена, если доступны
        if 'excel_feature_names' in model_stats:
            feature_names = model_stats['excel_feature_names']
        else:
            feature_names = model_stats['feature_names']
        
        # ИСПРАВЛЕНИЕ: Убедимся, что количество имен признаков соответствует количеству коэффициентов
        # Обрезаем список имен признаков, чтобы он соответствовал фактическому количеству коэффициентов
        feature_names = feature_names[:len(model_stats['coefficients'])]
        
        # Заполнение таблицы коэффициентов - только для фактических коэффициентов
        for i, name in enumerate(feature_names):
            if i < len(model_stats['coefficients']):
                # Определяем цветовой тег для строки на основе значимости
                is_significant = i < len(model_stats['p_values']) and model_stats['p_values'][i] < 0.05
                tag = "significant" if is_significant else "not_significant"
                
                # ИСПРАВЛЕНИЕ: Проверяем, что все необходимые массивы имеют достаточную длину
                coef_value = model_stats['coefficients'][i] if i < len(model_stats['coefficients']) else 0
                se_value = model_stats['se_coefficients'][i] if i < len(model_stats['se_coefficients']) else 0
                t_value = model_stats['t_values'][i] if i < len(model_stats['t_values']) else 0
                p_value = model_stats['p_values'][i] if i < len(model_stats['p_values']) else 1.0
                lower_ci_value = model_stats['lower_ci'][i] if i < len(model_stats['lower_ci']) else 0
                upper_ci_value = model_stats['upper_ci'][i] if i < len(model_stats['upper_ci']) else 0
                
                # Заполняем строку с данными
                coef_table.insert("", "end", values=(
                    name, 
                    f"{coef_value:.8f}", 
                    f"{se_value:.8f}", 
                    f"{t_value:.8f}", 
                    f"{p_value:.8f}",
                    f"{lower_ci_value:.8f}",
                    f"{upper_ci_value:.8f}"
                ), tags=(tag,))
        
        # Настраиваем цвета для тегов значимости
        coef_table.tag_configure("significant", background=DARK_THEME['success'])  # Темно-зеленый
        coef_table.tag_configure("not_significant", background=DARK_THEME['error'])  # Темно-красный    
        
        # Применяем стили темной темы к таблице
        style_treeview_tags(coef_table)
        
        # Горизонтальная прокрутка для таблицы коэффициентов
        scrollbar = ttk.Scrollbar(coef_frame, orient="horizontal", command=coef_table.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        coef_table.configure(xscrollcommand=scrollbar.set)
        
        coef_table.pack(fill=tk.X, padx=5, pady=5)
        
        # Если коэффициентов много, добавляем кнопку для просмотра всех в отдельном окне
        if len(model_stats['coefficients']) > 6:
            # Кнопка для просмотра всех коэффициентов
            view_all_button = tk.Button(
                coef_frame, 
                text="Просмотреть все коэффициенты",
                command=lambda: self.show_all_coefficients(model_stats),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            view_all_button.pack(pady=5)
    
    def show_all_coefficients(self, model_stats):
        """
        Показывает все коэффициенты модели в отдельном окне.
        
        Parameters:
        model_stats (dict): Статистические показатели модели
        """
        CoefficientViewer(self.parent, model_stats)

    def export_current_model_plots(self):
        """Экспортирует графики текущей модели в PDF."""
        X = getattr(self, f'X_{self.current_model}')
        y = self.y
        model = self.models[self.current_model]
        y_pred = self.predictions[self.current_model]
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF файлы", "*.pdf"), ("Все файлы", "*.*")],
            title=f"Экспортировать графики модели '{self.current_model}'"
        )
        
        if file_path:
            export_all_plots(self.df, X, y, model, y_pred, self.current_model, file_path)
    
    def export_report(self):
        """Экспортирует полный отчет о результатах анализа в PDF."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF файлы", "*.pdf"), ("Все файлы", "*.*")],
            title="Экспортировать отчет"
        )
        
        if not file_path:
            return
        
        try:
            # Получаем данные текущей модели
            model_stats = self.stats_dict[self.current_model]
            
            # Создаем отчет
            export_to_pdf(
                data=model_stats,
                filename=file_path,
                title=f"Отчет о регрессионном анализе - {self.current_model}",
                report_type="detailed",
                open_after_save=True
            )
            
            messagebox.showinfo("Экспорт", f"Отчет успешно экспортирован в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте отчета:\n{str(e)}")
    
    def show_graph(self, graph_index):
        """
        Показывает график в отдельном окне.
        
        Parameters:
        graph_index (int): Индекс типа графика (0-3)
        """
        # Закрываем предыдущее окно с графиком, если оно открыто
        if self.graph_window and self.graph_window.winfo_exists():
            self.graph_window.destroy()
        
        # Получаем данные для текущей модели
        X = getattr(self, f'X_{self.current_model}')
        y = self.y
        model = self.models[self.current_model]
        y_pred = self.predictions[self.current_model]
        
        # Создаем новое окно с графиком
        GraphViewer(self.parent, graph_index, self.df, X, y, model, y_pred, self.current_model)
    
    def populate_comparison_table(self):
        """Заполняет таблицу сравнения моделей расширенным набором показателей."""
        # Очищаем таблицу
        for i in self.comparison_table.get_children():
            self.comparison_table.delete(i)
            
        # Применяем стили темной темы к таблице сравнения
        style_treeview_tags(self.comparison_table)
        
        # Расширенный список показателей для сравнения
        metrics = [
            # Основные показатели качества модели
            ("R-квадрат", "r2"),
            ("Скорректированный R-квадрат", "adjusted_r2"),
            ("Множественный R", "multiple_r"),
            ("Стандартная ошибка", "se_regression"),
            
            # Показатели дисперсионного анализа
            ("F-статистика", "f_statistic"),
            ("Значимость F", "p_value_f"),
            
            # Степени свободы
            ("Степени свободы регрессии", "df_regression"),
            ("Степени свободы остатков", "df_residual"),
            ("Степени свободы всего", "df_total"),
            
            # Суммы квадратов
            ("Сумма квадратов регрессии", "ss_regression"),
            ("Сумма квадратов остатков", "ss_residual"),
            ("Общая сумма квадратов", "ss_total"),
            
            # Дополнительные статистические показатели
            ("Средний квадрат регрессии", "ms_regression"),
            ("Средний квадрат остатков", "ms_residual"),
            
            # Информация о наблюдениях
            ("Количество наблюдений", "observations")
        ]
        
        # Заполняем таблицу основными статистическими показателями
        for metric_name, metric_key in metrics:
            row_values = [metric_name]
            
            # Добавляем значения для каждой модели
            for model_type in ['all_groups', 'unemployed', 'combined']:
                if model_type in self.stats_dict:
                    value = self.stats_dict[model_type].get(metric_key, "—")
                    if isinstance(value, (int, float)):
                        # Форматируем вывод в зависимости от типа метрики
                        if metric_key in ["r2", "adjusted_r2", "multiple_r", "p_value_f"]:
                            row_values.append(f"{value:.6f}")
                        elif metric_key in ["df_regression", "df_residual", "df_total", "observations"]:
                            row_values.append(f"{int(value)}")
                        else:
                            row_values.append(f"{value:.2f}")
                    else:
                        row_values.append(str(value))
                else:
                    row_values.append("—")
            
            self.comparison_table.insert("", "end", values=tuple(row_values))
        
        # Добавляем информацию о коэффициентах моделей
        coef_info = [
            # Информация о количестве и значимости коэффициентов
            ("Количество коэффициентов (без константы)", None),
            ("Количество значимых коэффициентов (p < 0.05)", None),
            ("Процент значимых коэффициентов", None)
        ]
        
        for metric_name, metric_key in coef_info:
            row_values = [metric_name]
            
            for model_type in ['all_groups', 'unemployed', 'combined']:
                if model_type in self.stats_dict:
                    if metric_key is not None and metric_key in self.stats_dict[model_type]:
                        # Используем готовое значение из stats_dict
                        value = self.stats_dict[model_type][metric_key]
                        row_values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
                    else:
                        # Вычисляем значение
                        if metric_name == "Количество коэффициентов (без константы)":
                            # Вычисляем количество коэффициентов без константы
                            count = len(self.stats_dict[model_type].get('coefficients', [])) - 1
                            row_values.append(str(count))
                        elif metric_name == "Количество значимых коэффициентов (p < 0.05)":
                            # Считаем количество значимых коэффициентов (исключая константу)
                            p_values = self.stats_dict[model_type].get('p_values', [])
                            sig_count = sum(1 for i, p in enumerate(p_values) if i > 0 and p < 0.05)
                            row_values.append(str(sig_count))
                        elif metric_name == "Процент значимых коэффициентов":
                            # Вычисляем процент значимых коэффициентов
                            p_values = self.stats_dict[model_type].get('p_values', [])
                            total_count = len(p_values) - 1  # без константы
                            if total_count > 0:
                                sig_count = sum(1 for i, p in enumerate(p_values) if i > 0 and p < 0.05)
                                percent = (sig_count / total_count) * 100
                                row_values.append(f"{percent:.1f}%")
                            else:
                                row_values.append("—")
                else:
                    row_values.append("—")
            
            self.comparison_table.insert("", "end", values=tuple(row_values))
        
        # Дополнительные практические метрики для выбора модели
        practical_metrics = [
            ("Лучшая модель по R²", None),
            ("Лучшая модель по скорр. R²", None),
            ("Лучшая модель по F-статистике", None)
        ]
        
        # Находим и отмечаем лучшие модели по каждому критерию
        for metric_name, _ in practical_metrics:
            row_values = [metric_name]
            
            if metric_name == "Лучшая модель по R²":
                # Находим модель с наибольшим R²
                models_to_check = [m for m in ['all_groups', 'unemployed', 'combined'] if m in self.stats_dict]
                if models_to_check:
                    best_model = max(
                        models_to_check,
                        key=lambda m: self.stats_dict[m].get('r2', 0)
                    )
                    
                    for model_type in ['all_groups', 'unemployed', 'combined']:
                        if model_type == best_model:
                            row_values.append("✓")
                        else:
                            row_values.append("")
                else:
                    row_values.extend(["", "", ""])
                    
            elif metric_name == "Лучшая модель по скорр. R²":
                # Находим модель с наибольшим скорректированным R²
                models_to_check = [m for m in ['all_groups', 'unemployed', 'combined'] if m in self.stats_dict]
                if models_to_check:
                    best_model = max(
                        models_to_check,
                        key=lambda m: self.stats_dict[m].get('adjusted_r2', 0)
                    )
                    
                    for model_type in ['all_groups', 'unemployed', 'combined']:
                        if model_type == best_model:
                            row_values.append("✓")
                        else:
                            row_values.append("")
                else:
                    row_values.extend(["", "", ""])
                    
            elif metric_name == "Лучшая модель по F-статистике":
                # Находим модель с наибольшей F-статистикой
                models_to_check = [m for m in ['all_groups', 'unemployed', 'combined'] if m in self.stats_dict]
                if models_to_check:
                    best_model = max(
                        models_to_check,
                        key=lambda m: self.stats_dict[m].get('f_statistic', 0)
                    )
                    
                    for model_type in ['all_groups', 'unemployed', 'combined']:
                        if model_type == best_model:
                            row_values.append("✓")
                        else:
                            row_values.append("")
                else:
                    row_values.extend(["", "", ""])
            
            self.comparison_table.insert("", "end", values=tuple(row_values))
    
    def show_graphical_comparison(self):
        """Показывает графическое сравнение моделей."""
        # Импортируем настройки темы
        from ui.components.theme_manager import DARK_THEME, apply_chart_style
        
        # Создаем новое окно для сравнения
        comparison_window = tk.Toplevel(self.parent)
        comparison_window.title("Графическое сравнение моделей")
        comparison_window.geometry("800x600")
        comparison_window.configure(bg=DARK_THEME['primary'])
        
        # Делаем окно растягиваемым
        comparison_window.resizable(True, True)
        
        # Центрируем окно
        from ui.components.ui_helpers import center_window
        center_window(comparison_window, 800, 600)
        
        # Создаем фрейм для графика
        frame = tk.Frame(comparison_window, padx=10, pady=10, bg=DARK_THEME['primary'])
        frame.pack(fill=tk.BOTH, expand=True)
        
        try:
            # Импортируем необходимые модули
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            
            # Применяем стиль темной темы к графику
            apply_chart_style(plt)
            
            # Создаем график для сравнения предсказаний моделей
            fig = Figure(figsize=(10, 6), dpi=100, facecolor=DARK_THEME['primary'])
            ax = fig.add_subplot(111)
            ax.set_facecolor(DARK_THEME['bg'])
            
            # Получаем данные для оси X (годы)
            years = self.df['Год'] if 'Год' in self.df.columns else range(len(self.y))
            
            # Строим график фактических значений
            ax.plot(years, self.y, marker='o', color='black', linewidth=2, 
                markersize=6, label='Фактический ВВП')
            
            # Цвета для разных моделей и их русские названия
            colors = {
                'all_groups': '#3366cc',
                'unemployed': '#dc3545',
                'combined': '#28a745'
            }
            
            # Словарь с русскими названиями моделей
            model_names = {
                'all_groups': 'Модель от численности рабочих',
                'unemployed': 'Модель от безработицы',
                'combined': 'Комбинированная модель'
            }
            
            # Строим графики предсказаний для всех доступных моделей с русскими названиями
            for model_type, color in colors.items():
                if model_type in self.predictions:
                    ax.plot(years, self.predictions[model_type], linestyle='--', 
                        color=color, linewidth=2, label=model_names[model_type])
            
            # Настраиваем график
            ax.set_title('Сравнение предсказаний моделей', fontsize=14, pad=20, color=DARK_THEME['neutral'])
            ax.set_xlabel('Год', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
            ax.set_ylabel('ВВП', fontsize=12, labelpad=10, color=DARK_THEME['neutral'])
            
            # Настраиваем легенду
            legend = ax.legend(loc='best')
            legend.get_frame().set_facecolor(DARK_THEME['bg_light'])
            legend.get_frame().set_edgecolor(DARK_THEME['border'])
            
            for text in legend.get_texts():
                text.set_color(DARK_THEME['neutral'])
            
            # Настраиваем цвет делений и линий сетки
            ax.tick_params(colors=DARK_THEME['neutral'])
            ax.grid(True, alpha=0.5, color=DARK_THEME['bg_light'], linestyle='--')
            
            # Настраиваем цвет рамки
            for spine in ax.spines.values():
                spine.set_color(DARK_THEME['neutral'])
            
            # Удаляем лишние рамки
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Подгоняем макет
            fig.tight_layout()
            
            # Встраиваем график в окно
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Добавляем панель инструментов с темным стилем
            toolbar_frame = tk.Frame(frame, bg=DARK_THEME['primary'])
            toolbar_frame.pack(fill=tk.X)
            
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            toolbar.config(background=DARK_THEME['primary'])
            
            # Изменяем цвет всех кнопок на панели инструментов
            for button in toolbar.winfo_children():
                if isinstance(button, tk.Button):
                    button.config(
                        bg=DARK_THEME['bg_light'],
                        fg=DARK_THEME['neutral'],
                        activebackground=DARK_THEME['accent'],
                        activeforeground=DARK_THEME['text_light']
                    )
            
            # Добавляем таблицу качества моделей
            table_frame = tk.Frame(comparison_window, padx=10, pady=5, bg=DARK_THEME['primary'])
            table_frame.pack(fill=tk.X)
            
            tk.Label(table_frame, text="Качество моделей:", 
                    font=("Arial", 12, "bold"), 
                    bg=DARK_THEME['primary'],
                    fg=DARK_THEME['neutral']).pack(anchor=tk.W, pady=5)
            
            # Создаем таблицу
            table = ttk.Treeview(
                table_frame, 
                columns=("1", "2", "3", "4"), 
                show="headings", 
                height=3
            )
            
            table.heading("1", text="Метрика")
            table.heading("2", text="Модель от численности рабочих")
            table.heading("3", text="Модель от безработицы")
            table.heading("4", text="Комбинированная модель")
            
            table.column("1", width=150, anchor=tk.W)
            table.column("2", width=200, anchor=tk.CENTER)
            table.column("3", width=200, anchor=tk.CENTER)
            table.column("4", width=200, anchor=tk.CENTER)
            
            # Применяем стили темной темы к таблице
            style_treeview_tags(table)
            
            # Заполняем таблицу
            metrics = [
                ("R-квадрат", "r2"),
                ("Скорректированный R-квадрат", "adjusted_r2"),
                ("Стандартная ошибка", "se_regression")
            ]
            
            for metric_name, metric_key in metrics:
                row_values = [metric_name]
                
                for model_type in ['all_groups', 'unemployed', 'combined']:
                    if model_type in self.stats_dict:
                        value = self.stats_dict[model_type].get(metric_key, "—")
                        if isinstance(value, (int, float)):
                            row_values.append(f"{value:.6f}")
                        else:
                            row_values.append(str(value))
                    else:
                        row_values.append("—")
                
                table.insert("", "end", values=tuple(row_values))
            
            table.pack(fill=tk.X, padx=5, pady=5)
            
            # Кнопка закрытия
            button_frame = tk.Frame(comparison_window, pady=5, bg=DARK_THEME['primary'])
            button_frame.pack(fill=tk.X)
            
            close_button = tk.Button(
                button_frame, 
                text="Закрыть", 
                command=comparison_window.destroy,
                font=("Arial", 12),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            close_button.pack(side=tk.RIGHT, padx=10)
            
            # Кнопка экспорта
            export_button = tk.Button(
                button_frame, 
                text="Экспортировать сравнение", 
                command=lambda: self.export_comparison(fig),
                font=("Arial", 12),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            )
            export_button.pack(side=tk.LEFT, padx=10)
            
        except Exception as e:
            # В случае ошибки показываем сообщение
            tk.Label(frame, text=f"Ошибка при создании графика: {str(e)}", 
                    font=("Arial", 12), fg=DARK_THEME['error'],
                    bg=DARK_THEME['primary']).pack(pady=20)
            
            # Кнопка закрытия
            tk.Button(
                frame, 
                text="Закрыть", 
                command=comparison_window.destroy,
                font=("Arial", 12),
                bg=DARK_THEME['bg_light'],
                fg=DARK_THEME['neutral'],
                activebackground=DARK_THEME['accent'],
                activeforeground=DARK_THEME['text_light']
            ).pack(pady=10)
            
    def export_comparison(self, fig):
        """
        Экспортирует сравнение моделей в PDF.
        
        Parameters:
        fig (Figure): Объект фигуры matplotlib
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF файлы", "*.pdf"), ("Все файлы", "*.*")],
            title="Экспортировать сравнение моделей"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Экспорт", f"Сравнение моделей экспортировано в файл:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при экспорте сравнения:\n{str(e)}")
                
    def handle_back_button(self):
        """Handle the back button press by properly cleaning up before calling back_callback."""
        # First destroy this frame properly
        self.destroy()
        
        # Then call the callback to return to file selection
        self.back_callback()