# src/application_app.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import os
import json
from neural_network import NeuralNetwork
from data_loader import DataLoader

class ApplicationApp(ttk.Frame):
    def __init__(self, master, carpeta_raiz, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

        self.carpeta_raiz = carpeta_raiz
        self.modelo_path = os.path.join(self.carpeta_raiz, "models", "modelo_neural.pkl")
        self.estadisticas_path = os.path.join(self.carpeta_raiz, "models", "estadisticas.pkl")  # Ruta para estadísticas
        self.data_loader = DataLoader(
            imagenes_guardadas_json_ruta=os.path.join(
                self.carpeta_raiz, "imagenes_procesadas", "imagenes_guardadas.json"
            ),
            image_size=(64, 64)  # Asegúrate de usar el mismo tamaño que en el entrenamiento
        )

        # Cargar el modelo entrenado
        self.nn = self.cargar_modelo()

        # Cargar las clases
        self.classes = self.cargar_clases()

        # Variables para almacenar la imagen actual y los kernels seleccionados
        self.imagen_actual = None
        self.imagen_procesada = None
        self.kernels_aplicados = []

        # Cargar los filtros y kernels
        self.filtros = self.cargar_filtros()
        self.filtro_nombres = list(self.filtros.keys())
        self.kernels = []
        self.kernel_vars = {}
        self.check_vars = []

        # Variable para el filtro seleccionado
        self.selected_filtro = tk.StringVar(value='none')

        # Ruta del archivo JSON de kernels
        self.ruta_json = os.path.join(self.carpeta_raiz, "data", "kernel.json")

        # Cargar los kernels
        try:
            self.cargar_kernels()
        except Exception as e:
            messagebox.showerror("Error al Cargar Kernels", f"Ocurrió un error al cargar los kernels:\n{e}")
            self.destroy()
            return

        # Construir la interfaz gráfica
        self.construir_interfaz()

    def cargar_modelo(self):
        """Carga el modelo entrenado desde el archivo."""
        if not os.path.exists(self.modelo_path):
            messagebox.showerror("Error", f"No se encontró el modelo entrenado en la ruta:\n{self.modelo_path}")
            return None
        else:
            nn = NeuralNetwork.load_model(self.modelo_path)
            return nn

    def cargar_clases(self):
        """Carga las clases desde el DataLoader."""
        _, _, classes = self.data_loader.load_data()
        return classes

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

    def cargar_filtros(self):
        """Define los filtros disponibles."""
        filtros = {
            "Ninguno": "none",
            "Escala de Grises": "grayscale",
            "Rojo": "red",
            "Verde": "green",
            "Azul": "blue",
            "Blanco": "white",
            "Negro": "black"
        }
        return filtros

    def construir_interfaz(self):
        """Construye la interfaz gráfica de usuario."""
        # Título de la sección
        lbl_title = ttk.Label(self, text="Aplicación del Modelo Entrenado", font=("Helvetica", 16))
        lbl_title.pack(pady=10)

        # Frame principal para organizar los elementos
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Botón para cargar la imagen
        btn_cargar_imagen = ttk.Button(main_frame, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar_imagen.pack(pady=5)

        # Frame para las imágenes y la predicción
        frame_imagenes = ttk.Frame(main_frame)
        frame_imagenes.pack(pady=10, fill=tk.BOTH, expand=True)

        # Canvas para mostrar la imagen original
        self.canvas_original = tk.Canvas(frame_imagenes, width=550, height=350, bg="gray")
        self.canvas_original.grid(row=0, column=0, padx=10, pady=5)

        # Canvas para mostrar la imagen procesada
        self.canvas_procesada = tk.Canvas(frame_imagenes, width=550, height=350, bg="gray")
        self.canvas_procesada.grid(row=0, column=1, padx=10, pady=5)

        # Etiqueta para mostrar el resultado
        self.lbl_resultado = ttk.Label(main_frame, text="", font=("Helvetica", 14))
        self.lbl_resultado.pack(pady=10)

        # Frame para controles
        frame_controles = ttk.Frame(main_frame)
        frame_controles.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para los filtros
        frame_filtros = ttk.LabelFrame(frame_controles, text="Seleccionar Filtro de Color", padding=10)
        frame_filtros.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Radiobuttons para filtros de color
        for nombre_filtro in self.filtro_nombres:
            rb = ttk.Radiobutton(
                frame_filtros,
                text=nombre_filtro,
                variable=self.selected_filtro,
                value=self.filtros[nombre_filtro],
                command=self.actualizar_imagen
            )
            rb.pack(anchor='w', padx=5, pady=2)

        # Frame para los kernels con scrollbar
        frame_kernels = ttk.LabelFrame(frame_controles, text="Seleccionar Kernels", padding=10)
        frame_kernels.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas para los kernels
        canvas_kernels = tk.Canvas(frame_kernels)
        scrollbar_kernels = ttk.Scrollbar(frame_kernels, orient="vertical", command=canvas_kernels.yview)
        self.frame_kernels_inner = ttk.Frame(canvas_kernels)

        self.frame_kernels_inner.bind(
            "<Configure>",
            lambda e: canvas_kernels.configure(
                scrollregion=canvas_kernels.bbox("all")
            )
        )

        canvas_kernels.create_window((0, 0), window=self.frame_kernels_inner, anchor="nw")
        canvas_kernels.configure(yscrollcommand=scrollbar_kernels.set)

        canvas_kernels.pack(side="left", fill="both", expand=True)
        scrollbar_kernels.pack(side="right", fill="y")

        # Crear los Checkbuttons para los kernels
        self.crear_checkbuttons(self.frame_kernels_inner)

    def crear_checkbuttons(self, parent):
        """Crea Checkbuttons para cada kernel disponible."""
        columnas = 2  # Número de columnas para organizar los kernels
        for idx, kernel in enumerate(self.kernels):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(
                parent, text=kernel['name'], variable=var, command=self.actualizar_imagen
            )
            fila = idx // columnas
            columna = idx % columnas
            chk.grid(row=fila, column=columna, sticky='w', padx=5, pady=2)
            self.check_vars.append(var)
            self.kernel_vars[kernel['name']] = var
            # Añadir tooltip para la descripción (si tienes una clase ToolTip implementada)
            # self.agregar_tooltip(chk, kernel.get('description', 'Sin descripción'))

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
            self.imagen_actual = Image.open(ruta_imagen).convert("RGB")  # Asegurarse de que está en RGB
            self.imagen_procesada = self.imagen_actual.copy()  # Iniciar la imagen procesada como una copia
            self.restablecer_kernels_y_filtro()
            self.actualizar_imagen()  # Llamar para mostrar la imagen cargada según el estado del filtro
        except Exception as e:
            messagebox.showerror("Error al Cargar Imagen", f"Ocurrió un error al cargar la imagen:\n{e}")

    def restablecer_kernels_y_filtro(self):
        """Restablece los kernels y filtro seleccionados a los usados en el entrenamiento."""
        # Obtener los kernels y filtro aplicados a las imágenes en el JSON
        imagenes_json_ruta = os.path.join(
            self.carpeta_raiz, "imagenes_procesadas", "imagenes_guardadas.json"
        )
        if not os.path.exists(imagenes_json_ruta):
            messagebox.showerror("Error", f"No se encontró el archivo JSON de imágenes en la ruta especificada:\n{imagenes_json_ruta}")
            return

        with open(imagenes_json_ruta, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            messagebox.showerror("Error", "El archivo JSON de imágenes está vacío.")
            return

        # Asumimos que todas las imágenes tienen los mismos kernels y filtro aplicados
        kernels_usados = data[0].get('kernels_applied', [])
        filtro_usado = data[0].get('filter', 'none')

        # Restablecer los kernels seleccionados
        for kernel in self.kernels:
            nombre_kernel = kernel['name']
            if nombre_kernel in kernels_usados:
                self.kernel_vars[nombre_kernel].set(True)
            else:
                self.kernel_vars[nombre_kernel].set(False)

        # Restablecer el filtro seleccionado
        if filtro_usado in self.filtros.values():
            self.selected_filtro.set(filtro_usado)
        else:
            self.selected_filtro.set('none')

    def get_filtered_image(self):
        """Devuelve la imagen procesada con el filtro de color aplicado."""
        if self.imagen_actual is None:
            return None

        try:
            filtro = self.selected_filtro.get()
            imagen_filtrada = self.imagen_actual.copy()

            if filtro == 'grayscale':
                imagen_filtrada = imagen_filtrada.convert("L").convert("RGB")
            elif filtro == 'red':
                r, g, b = imagen_filtrada.split()
                imagen_filtrada = Image.merge("RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
            elif filtro == 'green':
                r, g, b = imagen_filtrada.split()
                imagen_filtrada = Image.merge("RGB", (Image.new("L", g.size), g, Image.new("L", g.size)))
            elif filtro == 'blue':
                r, g, b = imagen_filtrada.split()
                imagen_filtrada = Image.merge("RGB", (Image.new("L", b.size), Image.new("L", b.size), b))
            elif filtro == 'white':
                # Convertir a escala de grises y aplicar un umbral para resaltar las áreas blancas
                grayscale = imagen_filtrada.convert("L")
                threshold = 200  # Puedes ajustar este valor
                binary = grayscale.point(lambda x: 255 if x > threshold else 0, '1')
                imagen_filtrada = binary.convert("RGB")
            elif filtro == 'black':
                # Convertir a escala de grises y aplicar un umbral para resaltar las áreas negras
                grayscale = imagen_filtrada.convert("L")
                threshold = 50  # Puedes ajustar este valor
                binary = grayscale.point(lambda x: 0 if x < threshold else 255, '1')
                imagen_filtrada = binary.convert("RGB")
            # Si el filtro es 'none', no se aplica nada

            return imagen_filtrada
        except Exception as e:
            messagebox.showerror("Error al Aplicar Filtro de Color", f"Ocurrió un error al aplicar el filtro de color:\n{e}")
            return self.imagen_actual

    def actualizar_imagen(self):
        """Actualiza la imagen procesada en función del filtro de color y los kernels seleccionados."""
        if self.imagen_actual is None:
            return

        try:
            # Obtener la imagen filtrada
            self.imagen_procesada = self.get_filtered_image()

            # Aplicar los kernels seleccionados
            self.aplicar_kernels()

            # Mostrar la imagen original
            imagen_original_pil = self.redimensionar_imagen(self.imagen_actual, (550, 350))
            self.imagen_original_tk = ImageTk.PhotoImage(imagen_original_pil)
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.imagen_original_tk)
            self.canvas_original.update()

            # Mostrar la imagen procesada
            imagen_procesada_pil = self.redimensionar_imagen(self.imagen_procesada, (550, 350))
            self.imagen_procesada_tk = ImageTk.PhotoImage(imagen_procesada_pil)
            self.canvas_procesada.create_image(0, 0, anchor=tk.NW, image=self.imagen_procesada_tk)
            self.canvas_procesada.update()

            # Realizar la predicción
            self.procesar_y_clasificar_imagen()

        except Exception as e:
            messagebox.showerror("Error al Actualizar Imagen", f"Ocurrió un error al actualizar la imagen:\n{e}")

    def aplicar_kernels(self):
        """Aplica los kernels seleccionados a la imagen procesada."""
        if self.imagen_procesada is None:
            return

        # Obtener los kernels seleccionados
        seleccion = [var.get() for var in self.check_vars]
        kernels_seleccionados = [k for k, seleccionado in zip(self.kernels, seleccion) if seleccionado]
        self.kernels_aplicados = [k['name'] for k in kernels_seleccionados]

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
                return

            # Aplicar el filtro
            try:
                self.imagen_procesada = self.imagen_procesada.filter(filtro)
            except Exception as e:
                messagebox.showerror("Error al Aplicar Filtro", f"Ocurrió un error al aplicar el filtro de kernel '{kernel['name']}':\n{e}")
                return

    def redimensionar_imagen(self, imagen_pil, tamaño):
        """Redimensiona una imagen PIL manteniendo la relación de aspecto."""
        return imagen_pil.resize(tamaño, Image.LANCZOS)

    def procesar_y_clasificar_imagen(self):
        """Prepara la imagen para la predicción y realiza la clasificación."""
        if self.imagen_procesada is None or self.nn is None:
            return

        try:
            # Preparar la imagen para la predicción
            # Usar el mismo tamaño que durante el entrenamiento
            imagen_para_prediccion = self.imagen_procesada.resize((64, 64), Image.LANCZOS).convert("RGB")
            input_data = np.array(imagen_para_prediccion).reshape(1, -1).astype(np.float32)

            # Realizar la predicción
            prediction, confidence = self.nn.predict(input_data)
            clase_predicha = self.classes[prediction[0]]
            confianza = confidence[0] * 100

            # Mostrar el resultado
            self.lbl_resultado.config(text=f"Pez Predicho: {clase_predicha} ({confianza:.3f}% de confianza)")
        except Exception as e:
            messagebox.showerror("Error en Predicción", f"Ocurrió un error al realizar la predicción:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Aplicación de Clasificación de Peces")
    root.geometry("1200x800")
    app = ApplicationApp(root, carpeta_raiz=os.getcwd())
    root.mainloop()
