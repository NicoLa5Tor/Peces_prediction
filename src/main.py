# src/main.py

import tkinter as tk
from tkinter import ttk
import os
from image_processor import TratamientoFrame
from training_app import TrainingApp
from application_app import ApplicationApp

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicación de Clasificación de Peces")
        self.geometry("1200x800")

        # Carpeta raíz del proyecto
        self.carpeta_raiz = os.getcwd()

        # Crear un contenedor para los botones de navegación
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Crear un contenedor para las diferentes fases
        self.frame_contenedor = ttk.Frame(self)
        self.frame_contenedor.pack(fill=tk.BOTH, expand=True)

        # Crear los botones de navegación
        self.create_navigation_buttons()

        # Inicializar las variables para los frames
        self.tratamiento_frame = None
        self.entrenamiento_frame = None
        self.aplicacion_frame = None

        # Mostrar el frame inicial (Tratamiento de Imágenes)
        self.mostrar_tratamiento()

    def create_navigation_buttons(self):
        """Crea los botones para navegar entre las fases."""
        # Botón para Tratamiento de Imágenes
        btn_tratamiento = ttk.Button(
            self.button_frame,
            text="Tratamiento de Imágenes",
            command=self.mostrar_tratamiento
        )
        btn_tratamiento.pack(side=tk.LEFT, padx=5, pady=5)

        # Botón para Entrenamiento
        btn_entrenamiento = ttk.Button(
            self.button_frame,
            text="Entrenamiento",
            command=self.mostrar_entrenamiento
        )
        btn_entrenamiento.pack(side=tk.LEFT, padx=5, pady=5)

        # Botón para Aplicación
        btn_aplicacion = ttk.Button(
            self.button_frame,
            text="Aplicación",
            command=self.mostrar_aplicacion
        )
        btn_aplicacion.pack(side=tk.LEFT, padx=5, pady=5)

    def mostrar_tratamiento(self):
        """Muestra la fase de Tratamiento de Imágenes."""
        self.limpiar_frame_contenedor()
        self.tratamiento_frame = TratamientoFrame(
            self.frame_contenedor,
            carpeta_guardado=os.path.join(
                self.carpeta_raiz, "imagenes_procesadas"
            )
        )
        self.tratamiento_frame.pack(fill=tk.BOTH, expand=True)

    def mostrar_entrenamiento(self):
        """Muestra la fase de Entrenamiento."""
        self.limpiar_frame_contenedor()
        self.entrenamiento_frame = TrainingApp(
            self.frame_contenedor,
            carpeta_raiz=self.carpeta_raiz
        )
        self.entrenamiento_frame.pack(fill=tk.BOTH, expand=True)

    def mostrar_aplicacion(self):
        """Muestra la fase de Aplicación."""
        self.limpiar_frame_contenedor()
        self.aplicacion_frame = ApplicationApp(
            self.frame_contenedor,
            carpeta_raiz=self.carpeta_raiz
        )
        self.aplicacion_frame.pack(fill=tk.BOTH, expand=True)

    def limpiar_frame_contenedor(self):
        """Limpia el frame contenedor antes de mostrar una nueva fase."""
        for widget in self.frame_contenedor.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
