# src/application_app.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import json
from neural_network import NeuralNetwork
from data_loader import DataLoader
from scipy.signal import convolve2d

class ApplicationApp(ttk.Frame):
    def __init__(self, master, carpeta_raiz, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

        self.carpeta_raiz = carpeta_raiz
        self.modelo_path = os.path.join(self.carpeta_raiz, "models", "modelo_neural.pkl")
        self.data_loader = DataLoader(
            imagenes_guardadas_json_ruta=os.path.join(
                self.carpeta_raiz, "imagenes_procesadas", "imagenes_guardadas.json"
            ),
            image_size=(64, 64)
        )

        # Cargar el modelo entrenado
        self.nn = self.cargar_modelo()

        # Cargar las clases
        self.classes = self.cargar_clases()

        # Variables para almacenar la imagen actual y los kernels seleccionados
        self.imagen_actual = None

        # Título de la sección
        lbl_title = ttk.Label(self, text="Aplicación del Modelo Entrenado", font=("Helvetica", 16))
        lbl_title.pack(pady=10)

        # Botón para cargar la imagen
        btn_cargar_imagen = ttk.Button(self, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar_imagen.pack(pady=5)

        # Frame para los canvases y la predicción
        frame_imagenes = ttk.Frame(self)
        frame_imagenes.pack(pady=10)

        # Canvas para mostrar la imagen original
        self.canvas_original = tk.Canvas(frame_imagenes, width=300, height=300)
        self.canvas_original.grid(row=0, column=0, padx=10)

        # Canvas para mostrar la imagen procesada
        self.canvas_procesada = tk.Canvas(frame_imagenes, width=300, height=300)
        self.canvas_procesada.grid(row=0, column=1, padx=10)

        # Etiqueta para mostrar el resultado
        self.lbl_resultado = ttk.Label(self, text="", font=("Helvetica", 14))
        self.lbl_resultado.pack(pady=10)

        # Cargar los kernels
        self.kernels = self.cargar_kernels()
        self.kernel_nombres = list(self.kernels.keys())

        # Frame para los Checkbuttons de los kernels con scrollbar
        frame_kernels = ttk.LabelFrame(self, text="Seleccionar Kernels")
        frame_kernels.pack(pady=5, fill=tk.BOTH, expand=True)

        # Crear un canvas y scrollbar
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

        # Variables para los kernels seleccionados
        self.kernel_vars = {}
        columnas = 3  # Número de columnas para organizar los kernels
        for idx, nombre_kernel in enumerate(self.kernel_nombres):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(
                self.frame_kernels_inner,
                text=nombre_kernel,
                variable=var,
                command=self.kernels_cambiados
            )
            fila = idx // columnas
            columna = idx % columnas
            cb.grid(row=fila, column=columna, sticky='w', padx=5, pady=2)
            self.kernel_vars[nombre_kernel] = var

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
        ruta_json = os.path.join(self.carpeta_raiz, "data", "kernel.json")
        if not os.path.exists(ruta_json):
            messagebox.showerror("Error", f"No se encontró el archivo JSON de kernels en la ruta especificada:\n{ruta_json}")
            return {}

        with open(ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'kernels' not in data:
            messagebox.showerror("Error", "La clave 'kernels' no se encontró en el archivo JSON.")
            return {}

        kernels = data['kernels']
        if not kernels:
            messagebox.showerror("Error", "La lista de kernels está vacía en el archivo JSON.")
            return {}

        # Convertir los kernels a numpy arrays
        kernel_dict = {}
        for kernel in kernels:
            nombre = kernel['name']
            matriz = np.array(kernel['matrix'])
            kernel_dict[nombre] = matriz

        return kernel_dict

    def cargar_imagen(self):
        """Abre un diálogo para seleccionar una imagen y la procesa."""
        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if ruta_imagen:
            try:
                # Cargar la imagen original
                self.imagen_actual = Image.open(ruta_imagen).convert("RGB")
                self.mostrar_imagen_original()
                # Restablecer los kernels seleccionados
                self.restablecer_kernels()
                self.procesar_y_clasificar_imagen()
            except Exception as e:
                messagebox.showerror("Error", f"Ocurrió un error al procesar la imagen:\n{e}")

    def mostrar_imagen_original(self):
        """Muestra la imagen original en el canvas."""
        imagen_mostrada = self.imagen_actual.resize((300, 300), Image.LANCZOS)
        self.imagen_original_tk = ImageTk.PhotoImage(imagen_mostrada)
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.imagen_original_tk)
        self.canvas_original.update()

    def restablecer_kernels(self):
        """Restablece los kernels seleccionados a los usados en el entrenamiento."""
        # Obtener los kernels aplicados a las imágenes en el JSON
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

        # Asumimos que todas las imágenes tienen los mismos kernels aplicados
        kernels_usados = data[0].get('kernels_applied', [])
        # Restablecer los kernels seleccionados
        for nombre_kernel in self.kernel_vars:
            if nombre_kernel in kernels_usados:
                self.kernel_vars[nombre_kernel].set(True)
            else:
                self.kernel_vars[nombre_kernel].set(False)

    def procesar_y_clasificar_imagen(self):
        """Procesa la imagen con los kernels seleccionados y realiza la clasificación."""
        if self.imagen_actual is None:
            return

        # Obtener los kernels seleccionados
        kernels_seleccionados = [nombre for nombre, var in self.kernel_vars.items() if var.get()]
        if not kernels_seleccionados:
            messagebox.showwarning("Advertencia", "Por favor, selecciona al menos un kernel.")
            return

        imagen_procesada = self.imagen_actual.convert("L")  # Convertir a escala de grises
        for nombre_kernel in kernels_seleccionados:
            kernel_matriz = self.kernels.get(nombre_kernel)
            if kernel_matriz is not None:
                imagen_array = np.array(imagen_procesada)
                imagen_procesada_array = convolve2d(imagen_array, kernel_matriz, mode='same', boundary='fill', fillvalue=0)
                imagen_procesada_array = np.clip(imagen_procesada_array, 0, 255).astype(np.uint8)
                imagen_procesada = Image.fromarray(imagen_procesada_array)
            else:
                messagebox.showerror("Error", f"No se encontró el kernel '{nombre_kernel}'.")
                return

        # Mostrar la imagen procesada
        imagen_procesada_mostrada = imagen_procesada.resize((300, 300), Image.LANCZOS).convert("RGB")
        self.imagen_procesada_tk = ImageTk.PhotoImage(imagen_procesada_mostrada)
        self.canvas_procesada.create_image(0, 0, anchor=tk.NW, image=self.imagen_procesada_tk)
        self.canvas_procesada.update()

        # Preparar la imagen para la predicción
        input_data = self.data_loader.load_single_image(imagen_procesada.convert("RGB"))
        if input_data is None:
            messagebox.showerror("Error", "No se pudo procesar la imagen para la predicción.")
            return
        input_data = input_data.reshape(1, -1)

        # Realizar la predicción
        prediction, confidence = self.nn.predict(input_data)
        clase_predicha = self.classes[prediction[0]]
        confianza = confidence[0] * 100

        # Mostrar el resultado
        self.lbl_resultado.config(text=f"Pez Predicho: {clase_predicha} ({confianza:.3f}% de confianza)")

    def kernels_cambiados(self):
        """Se llama cuando se cambian los kernels seleccionados."""
        self.procesar_y_clasificar_imagen()
