# src/image_processor.py

import json
import os
import datetime
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from data_loader import DataLoader

class ToolTip:
    """
    Crea un tooltip para un widget.

    Uso:
        tooltip = ToolTip(widget, "Texto del tooltip")
    """
    def __init__(self, widget, texto):
        self.widget = widget
        self.texto = texto
        self.tooltip = None
        self.widget.bind("<Enter>", self.mostrar_tooltip)
        self.widget.bind("<Leave>", self.ocultar_tooltip)

    def mostrar_tooltip(self, event=None):
        if self.tooltip:
            return
        # Posicionar el tooltip cerca del widget
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)  # Sin bordes
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip, text=self.texto, background="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

    def ocultar_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TratamientoFrame(ttk.Frame):
    def __init__(self, master, carpeta_guardado, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master  # Referencia al contenedor principal
        self.carpeta_guardado = carpeta_guardado
        os.makedirs(self.carpeta_guardado, exist_ok=True)

        # Lista para almacenar información de imágenes guardadas
        self.imagenes_guardadas = []

        # Ruta del archivo JSON de kernels
        self.ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data", "kernel.json")

        # Variables para almacenar imágenes
        self.imagen_original = None
        self.imagen_procesada = None
        self.kernels = []
        self.check_vars = []

        # Variables para almacenar filtros y kernels aplicados
        self.kernels_aplicados = []

        # Variable para seleccionar el filtro de color
        self.filtro_color = tk.StringVar(value='none')  # Valores posibles: 'none', 'grayscale', 'red', 'green', 'blue', 'white', 'black'

        # Variable para seleccionar el tipo de pez
        self.tipo_pez = tk.StringVar(value='Ángel')  # Valores: 'Ángel', 'Trucha Arcoíris'

        # Instanciar DataLoader
        self.data_loader = DataLoader(
            imagenes_guardadas_json_ruta=os.path.join(self.carpeta_guardado, "imagenes_guardadas.json"),
            image_size=(100, 100)
        )

        # Cargar los kernels
        try:
            self.cargar_kernels()
        except Exception as e:
            messagebox.showerror("Error al Cargar Kernels", f"Ocurrió un error al cargar los kernels:\n{e}")
            self.master.destroy()
            return

        # Configuración de la GUI
        self.configurar_gui()

    def cargar_kernels(self):
        """Carga los kernels desde el archivo JSON."""
        if not os.path.exists(self.ruta_json):
            raise FileNotFoundError(f"No se encontró el archivo JSON en la ruta especificada:\n{self.ruta_json}")

        with open(self.ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'kernels' not in data:
            raise KeyError("La clave 'kernels' no se encontró en el archivo JSON.")

        self.kernels = data['kernels']
        if not self.kernels:
            raise ValueError("La lista de kernels está vacía en el archivo JSON.")
        print(f"Se cargaron {len(self.kernels)} kernels desde el archivo JSON.")

    # ... (El resto del código permanece igual)


    def configurar_gui(self):
        """Configura los elementos de la interfaz gráfica."""
        # Frame principal dividido en dos: imagen y controles
        frame_imagenes = ttk.Frame(self)
        frame_imagenes.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Subframes para imagen original y procesada, con tamaño fijo
        frame_original = ttk.LabelFrame(frame_imagenes, text="Imagen Original", padding=10, width=550, height=350)
        frame_original.pack(side=tk.LEFT, padx=5, pady=5)
        frame_original.pack_propagate(False)  # Desactivar propagación para mantener el tamaño fijo

        frame_procesada = ttk.LabelFrame(frame_imagenes, text="Imagen Procesada", padding=10, width=550, height=350)
        frame_procesada.pack(side=tk.RIGHT, padx=5, pady=5)
        frame_procesada.pack_propagate(False)  # Desactivar propagación para mantener el tamaño fijo

        # Labels para mostrar las imágenes con tamaño fijo
        self.label_original = tk.Label(frame_original, width=550, height=350, bg="gray")
        self.label_original.pack()

        self.label_procesada = tk.Label(frame_procesada, width=550, height=350, bg="gray")
        self.label_procesada.pack()

        # Frame para controles con scrollbar
        frame_controles = ttk.Frame(self)
        frame_controles.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas para los controles
        canvas_controles = tk.Canvas(frame_controles)
        scrollbar_controles = ttk.Scrollbar(frame_controles, orient="vertical", command=canvas_controles.yview)
        scrollable_frame = ttk.Frame(canvas_controles)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_controles.configure(
                scrollregion=canvas_controles.bbox("all")
            )
        )

        canvas_controles.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_controles.configure(yscrollcommand=scrollbar_controles.set)

        canvas_controles.pack(side="left", fill="both", expand=True)
        scrollbar_controles.pack(side="right", fill="y")

        self.bind_scroll_events(canvas_controles)
        # Bind scroll events to scrollable_frame
        scrollable_frame.bind("<MouseWheel>", lambda event: self.scroll_kernels(event, canvas_controles))
        scrollable_frame.bind("<Button-4>", lambda event: self.scroll_kernels(event, canvas_controles))
        scrollable_frame.bind("<Button-5>", lambda event: self.scroll_kernels(event, canvas_controles))

        # Crear una estructura de grid para aprovechar mejor el espacio
        # Dividir scrollable_frame en dos columnas: una para kernels y otra para controles
        frame_kernels = ttk.Frame(scrollable_frame)
        frame_kernels.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_controles_lateral = ttk.Frame(scrollable_frame)
        frame_controles_lateral.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=5)

        # *** Sección: Selección de Kernels en Dos Columnas ***
        # Crear dos subframes dentro de frame_kernels para 3x3 y 5x5
        frame_kernels_3x3 = ttk.LabelFrame(frame_kernels, text="Kernels 3x3", padding=5)
        frame_kernels_3x3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_kernels_5x5 = ttk.LabelFrame(frame_kernels, text="Kernels 5x5", padding=5)
        frame_kernels_5x5.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas y scrollbar para los Checkbuttons de 3x3
        canvas_3x3 = tk.Canvas(frame_kernels_3x3)
        scrollbar_y_3x3 = ttk.Scrollbar(frame_kernels_3x3, orient="vertical", command=canvas_3x3.yview)
        scrollable_frame_3x3 = ttk.Frame(canvas_3x3)

        scrollable_frame_3x3.bind(
            "<Configure>",
            lambda e: canvas_3x3.configure(
                scrollregion=canvas_3x3.bbox("all")
            )
        )

        canvas_3x3.create_window((0, 0), window=scrollable_frame_3x3, anchor="nw")
        canvas_3x3.configure(yscrollcommand=scrollbar_y_3x3.set)

        canvas_3x3.pack(side="left", fill="both", expand=True)
        scrollbar_y_3x3.pack(side="right", fill="y")

        self.bind_scroll_events(canvas_3x3)
        # Bind scroll events to scrollable_frame_3x3
        scrollable_frame_3x3.bind("<MouseWheel>", lambda event: self.scroll_kernels(event, canvas_3x3))
        scrollable_frame_3x3.bind("<Button-4>", lambda event: self.scroll_kernels(event, canvas_3x3))
        scrollable_frame_3x3.bind("<Button-5>", lambda event: self.scroll_kernels(event, canvas_3x3))

        # Crear los Checkbuttons (kernels) en el frame scrollable de 3x3
        self.crear_checkbuttons(scrollable_frame_3x3, size='3x3')

        # Canvas y scrollbar para los Checkbuttons de 5x5
        canvas_5x5 = tk.Canvas(frame_kernels_5x5)
        scrollbar_y_5x5 = ttk.Scrollbar(frame_kernels_5x5, orient="vertical", command=canvas_5x5.yview)
        scrollable_frame_5x5 = ttk.Frame(canvas_5x5)

        scrollable_frame_5x5.bind(
            "<Configure>",
            lambda e: canvas_5x5.configure(
                scrollregion=canvas_5x5.bbox("all")
            )
        )

        canvas_5x5.create_window((0, 0), window=scrollable_frame_5x5, anchor="nw")
        canvas_5x5.configure(yscrollcommand=scrollbar_y_5x5.set)

        canvas_5x5.pack(side="left", fill="both", expand=True)
        scrollbar_y_5x5.pack(side="right", fill="y")

        self.bind_scroll_events(canvas_5x5)
        # Bind scroll events to scrollable_frame_5x5
        scrollable_frame_5x5.bind("<MouseWheel>", lambda event: self.scroll_kernels(event, canvas_5x5))
        scrollable_frame_5x5.bind("<Button-4>", lambda event: self.scroll_kernels(event, canvas_5x5))
        scrollable_frame_5x5.bind("<Button-5>", lambda event: self.scroll_kernels(event, canvas_5x5))

        # Crear los Checkbuttons (kernels) en el frame scrollable de 5x5
        self.crear_checkbuttons(scrollable_frame_5x5, size='5x5')
        # *** Fin de la Sección ***

        # *** Sección: Controles Lateral (Botones y Radiobuttons) ***
        # Aquí se colocan los botones y radiobuttons en el espacio lateral
        # Botón para cargar la imagen procesada
        btn_cargar = ttk.Button(frame_controles_lateral, text="📂 Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(padx=5, pady=5, fill=tk.X)

        # Botón para guardar la imagen procesada
        btn_guardar = ttk.Button(frame_controles_lateral, text="💾 Guardar Imagen Procesada", command=self.guardar_imagen)
        btn_guardar.pack(padx=5, pady=10, fill=tk.X)

        # Botón para generar JSON
        btn_generar_json = ttk.Button(frame_controles_lateral, text="📄 Generar JSON", command=self.generar_json)
        btn_generar_json.pack(padx=5, pady=5, fill=tk.X)

        # *** Sección: Selección del Tipo de Pez ***
        frame_tipo_pez = ttk.LabelFrame(frame_controles_lateral, text="Seleccionar Tipo de Pez", padding=10)
        frame_tipo_pez.pack(fill=tk.X, padx=5, pady=5)

        # Radiobuttons para tipo de pez
        radiobtn_angel = ttk.Radiobutton(frame_tipo_pez, text="Ángel", variable=self.tipo_pez, value='Ángel')
        radiobtn_angel.pack(anchor='w', padx=5, pady=2)

        radiobtn_trucha = ttk.Radiobutton(frame_tipo_pez, text="Trucha Arcoíris", variable=self.tipo_pez, value='Trucha Arcoíris')
        radiobtn_trucha.pack(anchor='w', padx=5, pady=2)
        # *** Fin de la Sección ***

        # *** Sección: Selección de Filtros de Color con Radiobuttons ***
        frame_filtros = ttk.LabelFrame(scrollable_frame, text="Seleccionar Filtro de Color", padding=10)
        frame_filtros.pack(fill=tk.X, padx=5, pady=5)

        # Radiobuttons para filtros de color
        radiobtn_none = ttk.Radiobutton(frame_filtros, text="Ninguno", variable=self.filtro_color, value='none', command=self.actualizar_imagen)
        radiobtn_none.pack(anchor='w', padx=5, pady=2)

        radiobtn_grayscale = ttk.Radiobutton(frame_filtros, text="Escala de Grises", variable=self.filtro_color, value='grayscale', command=self.actualizar_imagen)
        radiobtn_grayscale.pack(anchor='w', padx=5, pady=2)

        radiobtn_red = ttk.Radiobutton(frame_filtros, text="Rojo", variable=self.filtro_color, value='red', command=self.actualizar_imagen)
        radiobtn_red.pack(anchor='w', padx=5, pady=2)

        radiobtn_green = ttk.Radiobutton(frame_filtros, text="Verde", variable=self.filtro_color, value='green', command=self.actualizar_imagen)
        radiobtn_green.pack(anchor='w', padx=5, pady=2)

        radiobtn_blue = ttk.Radiobutton(frame_filtros, text="Azul", variable=self.filtro_color, value='blue', command=self.actualizar_imagen)
        radiobtn_blue.pack(anchor='w', padx=5, pady=2)

        radiobtn_white = ttk.Radiobutton(frame_filtros, text="Blanco", variable=self.filtro_color, value='white', command=self.actualizar_imagen)
        radiobtn_white.pack(anchor='w', padx=5, pady=2)

        radiobtn_black = ttk.Radiobutton(frame_filtros, text="Negro", variable=self.filtro_color, value='black', command=self.actualizar_imagen)
        radiobtn_black.pack(anchor='w', padx=5, pady=2)
        # *** Fin de la Sección ***

        # Barra de estado
        self.barra_estado = ttk.Label(self.master, text="Listo", relief=tk.SUNKEN, anchor='w')
        self.barra_estado.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

    def bind_scroll_events(self, canvas):
        """Asocia eventos de scroll para diferentes sistemas operativos."""
        # Enlazar eventos de desplazamiento directamente al canvas
        # Para Windows y MacOS
        canvas.bind("<MouseWheel>", lambda event: self.scroll_kernels(event, canvas))
        # Para Linux
        canvas.bind("<Button-4>", lambda event: self.scroll_kernels(event, canvas))
        canvas.bind("<Button-5>", lambda event: self.scroll_kernels(event, canvas))

        # Enfocar el canvas al entrar con el ratón para recibir eventos de desplazamiento
        canvas.bind("<Enter>", lambda event: canvas.focus_set())

    def scroll_kernels(self, event, canvas):
        """Permite desplazarse en el frame de kernels utilizando el scroll del ratón."""
        try:
            if hasattr(event, 'delta'):
                # Para Windows y MacOS
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif event.num == 4:
                # Para Linux: scroll up
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                # Para Linux: scroll down
                canvas.yview_scroll(1, "units")
        except Exception as e:
            print(f"Error en scroll_kernels: {e}")

    def crear_checkbuttons(self, parent, size='3x3'):
        """Crea Checkbuttons para cada kernel disponible basado en el tamaño especificado."""
        # Filtrar kernels por el tamaño especificado
        kernels_filtrados = [k for k in self.kernels if k.get('size', '3x3') == size]

        if not kernels_filtrados:
            lbl_no_kernels = ttk.Label(parent, text=f"No hay kernels disponibles para el tamaño {size}.")
            lbl_no_kernels.pack(padx=10, pady=10)
            return

        for kernel in kernels_filtrados:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(parent, text=kernel['name'], variable=var, command=self.on_kernel_toggle)
            chk.pack(anchor='w', padx=5, pady=2)
            self.check_vars.append(var)
            # Añadir tooltip para la descripción
            self.agregar_tooltip(chk, kernel.get('description', 'Sin descripción'))

    def agregar_tooltip(self, widget, texto):
        """Añade un tooltip a un widget."""
        tooltip = ToolTip(widget, texto)

    def cargar_imagen(self):
        """Abre un diálogo para seleccionar y cargar una imagen."""
        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.gif"), ("Todos los Archivos", "*.*")]
        )
        if not ruta_imagen:
            return

        try:
            # Cargar la imagen en PIL
            self.imagen_original = Image.open(ruta_imagen).convert("RGB")  # Asegurarse de que está en RGB
            self.imagen_procesada = self.imagen_original.copy()  # Iniciar la imagen procesada como una copia
            self.kernels_aplicados = []  # Reiniciar la lista de kernels aplicados
            self.actualizar_imagen()  # Llamar para mostrar la imagen cargada según el estado del filtro
            self.barra_estado.config(text="Imagen cargada exitosamente.")
        except Exception as e:
            messagebox.showerror("Error al Cargar Imagen", f"Ocurrió un error al cargar la imagen:\n{e}")
            self.barra_estado.config(text="Error al cargar la imagen.")

    def get_filtered_image(self):
        """Devuelve la imagen procesada con el filtro de color aplicado."""
        if self.imagen_procesada is None:
            return None

        try:
            filtro = self.filtro_color.get()

            if filtro == 'grayscale':
                return self.imagen_procesada.convert("L").convert("RGB")
            elif filtro == 'red':
                r, g, b = self.imagen_procesada.split()
                return Image.merge("RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
            elif filtro == 'green':
                r, g, b = self.imagen_procesada.split()
                return Image.merge("RGB", (Image.new("L", g.size), g, Image.new("L", g.size)))
            elif filtro == 'blue':
                r, g, b = self.imagen_procesada.split()
                return Image.merge("RGB", (Image.new("L", b.size), Image.new("L", b.size), b))
            elif filtro == 'white':
                # Convertir a escala de grises y aplicar un umbral para resaltar las áreas blancas
                grayscale = self.imagen_procesada.convert("L")
                threshold = 200  # Puedes ajustar este valor
                binary = grayscale.point(lambda x: 255 if x > threshold else 0, '1')
                return binary.convert("RGB")
            elif filtro == 'black':
                # Convertir a escala de grises y aplicar un umbral para resaltar las áreas negras
                grayscale = self.imagen_procesada.convert("L")
                threshold = 50  # Puedes ajustar este valor
                binary = grayscale.point(lambda x: 0 if x < threshold else 255, '1')
                return binary.convert("RGB")
            else:
                # Ningún filtro aplicado
                return self.imagen_procesada
        except Exception as e:
            messagebox.showerror("Error al Aplicar Filtro de Color", f"Ocurrió un error al aplicar el filtro de color:\n{e}")
            self.barra_estado.config(text="Error al aplicar el filtro de color.")
            return self.imagen_procesada

    def actualizar_imagen(self, value=None):
        """Actualiza la imagen procesada en función del filtro de color seleccionado."""
        if self.imagen_original is None:
            return

        try:
            # Mostrar la imagen original sin cambios
            imagen_original_pil = self.redimensionar_imagen(self.imagen_original, (500, 350))
            imagen_original_tk = ImageTk.PhotoImage(imagen_original_pil)
            self.label_original.config(image=imagen_original_tk)
            self.label_original.image = imagen_original_tk  # Mantener una referencia

            # Obtener la imagen procesada con el filtro de color aplicado
            imagen_procesada_mostrar = self.get_filtered_image()

            # Redimensionar y mostrar la imagen procesada
            imagen_procesada_pil = self.redimensionar_imagen(imagen_procesada_mostrar, (500, 350))
            imagen_procesada_tk = ImageTk.PhotoImage(imagen_procesada_pil)
            self.label_procesada.config(image=imagen_procesada_tk)
            self.label_procesada.image = imagen_procesada_tk  # Mantener una referencia

        except Exception as e:
            messagebox.showerror("Error al Actualizar Imagen", f"Ocurrió un error al actualizar la imagen:\n{e}")
            self.barra_estado.config(text="Error al actualizar la imagen.")

    def redimensionar_imagen(self, imagen_pil, tamaño):
        """Redimensiona una imagen PIL manteniendo la relación de aspecto."""
        return imagen_pil.resize(tamaño, Image.LANCZOS)

    def on_kernel_toggle(self):
        """Callback para cuando se selecciona o deselecciona un kernel."""
        if self.imagen_original is None:
            messagebox.showwarning("Sin Imagen", "Por favor, carga una imagen primero.")
            self.barra_estado.config(text="Intento de aplicar kernels sin cargar una imagen.")
            return  # No hay imagen cargada

        # Aplicar todos los kernels seleccionados
        seleccion = [var.get() for var in self.check_vars]
        kernels_seleccionados = [k for k, seleccionado in zip(self.kernels, seleccion) if seleccionado]

        imagen_procesada = self.imagen_original.copy()
        nombres_aplicados = []

        for kernel in kernels_seleccionados:
            matriz = kernel['matrix']

            # Convertir la matriz a una lista plana
            kernel_flat = [item for sublist in matriz for item in sublist]

            # Tamaño del kernel
            size = kernel.get('size', '3x3')  # Usar 3x3 por defecto si no está especificado
            if 'x' in size.lower():
                ancho, alto = map(int, size.lower().split('x'))
            else:
                ancho, alto = 3, 3  # Valor por defecto

            # Crear el filtro de kernel
            try:
                filtro = ImageFilter.Kernel(
                    size=(ancho, alto),
                    kernel=kernel_flat,
                    scale=sum(kernel_flat) if sum(kernel_flat) != 0 else 1,
                    offset=0
                )
            except Exception as e:
                messagebox.showerror("Error al Crear Filtro", f"Ocurrió un error al crear el filtro de kernel '{kernel['name']}':\n{e}")
                self.barra_estado.config(text="Error al crear el filtro de kernel.")
                return

            # Aplicar el filtro
            try:
                imagen_procesada = imagen_procesada.filter(filtro)
                nombres_aplicados.append(kernel['name'])
            except Exception as e:
                messagebox.showerror("Error al Aplicar Filtro", f"Ocurrió un error al aplicar el filtro de kernel '{kernel['name']}':\n{e}")
                self.barra_estado.config(text="Error al aplicar el filtro de kernel.")
                return

        # Guardar la imagen procesada
        self.imagen_procesada = imagen_procesada
        self.kernels_aplicados = nombres_aplicados  # Actualizar la lista de kernels aplicados
        self.actualizar_imagen()  # Actualizar la imagen en la interfaz
        self.barra_estado.config(text=f"Kernels aplicados: {', '.join(nombres_aplicados)}.")

    def guardar_imagen(self):
        """Guarda la imagen procesada en una carpeta específica con detalles de filtros y kernels aplicados."""
        if self.imagen_procesada is None:
            messagebox.showwarning("Sin Imagen Procesada", "Por favor, aplica al menos un kernel antes de guardar.")
            self.barra_estado.config(text="Intento de guardar una imagen sin procesar.")
            return

        try:
            # Obtener la imagen procesada con el filtro de color aplicado
            imagen_guardar = self.get_filtered_image()

            # Redimensionar la imagen a 100x100 píxeles
            imagen_guardar_resized = imagen_guardar.resize((100, 100), Image.LANCZOS)

            # Generar un nombre único para la imagen
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"imagen_procesada_{timestamp}.png"
            ruta_guardado = os.path.join(self.carpeta_guardado, nombre_archivo)

            # Guardar la imagen redimensionada
            imagen_guardar_resized.save(ruta_guardado)

            # Añadir la información de la imagen guardada a la lista
            self.imagenes_guardadas.append({
                'name': nombre_archivo,
                'path': ruta_guardado,
                'filter': self.filtro_color.get(),
                'kernels_applied': self.kernels_aplicados.copy(),
                'tipo_pez': self.tipo_pez.get()
            })

            # Guardar la lista en el archivo JSON
            with open(os.path.join(self.carpeta_guardado, "imagenes_guardadas.json"), 'w', encoding='utf-8') as f:
                json.dump(self.imagenes_guardadas, f, ensure_ascii=False, indent=4)

            messagebox.showinfo("Éxito al Guardar", f"Imagen procesada guardada exitosamente en:\n{ruta_guardado}")
            self.barra_estado.config(text=f"Imagen guardada en: {ruta_guardado}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"Ocurrió un error al guardar la imagen:\n{e}")
            self.barra_estado.config(text="Error al guardar la imagen.")

    def generar_json(self):
        """Genera un archivo JSON con el nombre, ruta, filtros, kernels aplicados y tipo de pez de las imágenes guardadas."""
        if not self.imagenes_guardadas:
            messagebox.showwarning("Sin Imágenes Guardadas", "No hay imágenes guardadas para generar el JSON.")
            self.barra_estado.config(text="Intento de generar JSON sin imágenes guardadas.")
            return

        try:
            ruta_json = os.path.join(self.carpeta_guardado, "imagenes_guardadas.json")
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(self.imagenes_guardadas, f, ensure_ascii=False, indent=4)

            messagebox.showinfo("JSON Generado", f"Archivo JSON generado exitosamente en:\n{ruta_json}")
            self.barra_estado.config(text=f"JSON generado en: {ruta_json}")
        except Exception as e:
            messagebox.showerror("Error al Generar JSON", f"Ocurrió un error al generar el JSON:\n{e}")
            self.barra_estado.config(text="Error al generar el JSON.")
