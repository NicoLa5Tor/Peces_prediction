# src/training_app.py

import tkinter as tk
from tkinter import ttk, messagebox
from neural_network import NeuralNetwork
from data_loader import DataLoader
import threading
import os
import numpy as np
import queue
import time
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class TrainingApp(ttk.Frame):
    def __init__(self, master, carpeta_raiz, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)
        
        self.carpeta_raiz = carpeta_raiz
        self.models_dir = os.path.join(self.carpeta_raiz, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.queue = queue.Queue()
        self.max_epochs = None  # No hay límite de épocas

        # Almacenar las pérdidas para la gráfica
        self.losses = []

        # Título de la sección
        lbl_title = ttk.Label(self, text="Entrenamiento de la Red Neuronal", font=("Helvetica", 16))
        lbl_title.pack(pady=10)

        # Frame para los parámetros de entrenamiento
        frame_parametros = ttk.LabelFrame(self, text="Parámetros de Entrenamiento", padding=10)
        frame_parametros.pack(padx=20, pady=10, fill=tk.X)

        # Parámetros del entrenamiento
        self.agregar_parametro(frame_parametros, "Número de Neuronas en Capa Oculta:", 0, "64")
        self.agregar_parametro(frame_parametros, "Tasa de Aprendizaje (Alpha):", 1, "0.001")
        self.agregar_parametro(frame_parametros, "Error Deseado:", 2, "0.001")

        # Botón para iniciar el entrenamiento
        btn_train = ttk.Button(self, text="Iniciar Entrenamiento", command=self.start_training)
        btn_train.pack(pady=10)

        # Barra de progreso y estado
        self.progress = ttk.Progressbar(self, orient='horizontal', mode='indeterminate', length=400)
        self.progress.pack(pady=10)
        self.status_label = ttk.Label(self, text="Esperando para iniciar el entrenamiento.")
        self.status_label.pack(pady=5)

        # PanedWindow para dividir la consola y la gráfica
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame para la consola y la gráfica
        frame_consola_grafica = ttk.Frame(paned)
        paned.add(frame_consola_grafica, weight=1)

        # Frame para la consola
        frame_consola = ttk.Frame(frame_consola_grafica)
        frame_consola.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Text widget para mostrar la salida del entrenamiento
        self.text_output = tk.Text(frame_consola, height=10, wrap='word')
        self.text_output.pack(fill=tk.BOTH, expand=True)
        self.text_output.configure(font=("Courier", 10))

        # Frame para la gráfica
        frame_grafica = ttk.Frame(frame_consola_grafica)
        frame_grafica.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        # Ajustar pesos para distribuir el espacio (40% consola, 60% gráfica)
        frame_consola_grafica.rowconfigure(0, weight=2)  # Consola
        frame_consola_grafica.rowconfigure(1, weight=3)  # Gráfica
        frame_consola_grafica.columnconfigure(0, weight=1)

        # Crear la figura de matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Error vs. Épocas")
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Error")

        # Embebemos la figura en Tkinter
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=frame_grafica)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Iniciar el procesamiento de la cola
        self.after(100, self.process_queue)

    def agregar_parametro(self, frame, texto, fila, valor_default):
        """Agrega un campo de entrada de parámetros al frame."""
        label = ttk.Label(frame, text=texto)
        label.grid(row=fila, column=0, padx=5, pady=5, sticky='w')
        entry = ttk.Entry(frame)
        entry.insert(0, valor_default)
        entry.grid(row=fila, column=1, padx=5, pady=5, sticky='w')
        setattr(self, f"entry_{fila}", entry)

    def start_training(self):
        """Inicia el proceso de entrenamiento."""
        try:
            hidden_size = int(self.entry_0.get())
            learning_rate = float(self.entry_1.get())
            desired_error = float(self.entry_2.get())
        except ValueError:
            messagebox.showerror("Entrada Inválida", "Por favor ingresa valores numéricos válidos.")
            return

        # Cargar los datos para el entrenamiento
        self.status_label.config(text="Estado: Cargando datos...")
        try:
            data_loader = DataLoader(
                imagenes_guardadas_json_ruta=os.path.join(
                    self.carpeta_raiz, "imagenes_procesadas", "imagenes_guardadas.json"
                ),
                image_size=(64, 64),
                augment_data=True  # Activar Data Augmentation
            )
            inputs, labels, classes = data_loader.load_data()
            self.data_loader = data_loader  # Guardar para uso posterior
            print(f"Datos cargados: {inputs.shape[0]} muestras.")
        except Exception as e:
            messagebox.showerror("Error al Cargar Datos", f"Ocurrió un error al cargar los datos:\n{e}")
            self.status_label.config(text="Estado: Error al cargar los datos.")
            return

        if inputs.size == 0 or labels.size == 0:
            messagebox.showwarning("Datos Insuficientes", "No hay datos de entrenamiento disponibles.")
            self.status_label.config(text="Estado: Datos insuficientes.")
            return

        # Normalizar los datos
        self.mean = self.data_loader.mean
        self.std = self.data_loader.std
        inputs = (inputs - self.mean) / self.std

        # Dividir los datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42, stratify=labels)

        # Crear la red neuronal
        input_size = inputs.shape[1]
        output_size = len(classes)
        nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        print(f"Red Neuronal creada: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, learning_rate={learning_rate}")

        # Configurar la barra de progreso
        self.progress.start()
        self.status_label.config(text="Estado: Entrenando la red neuronal...")

        # Limpiar el texto de salida
        self.text_output.delete('1.0', tk.END)

        # Limpiar las pérdidas anteriores
        self.losses.clear()

        # Entrenar la red neuronal en un hilo separado
        threading.Thread(target=self.train_nn, args=(nn, X_train, y_train, X_val, y_val, classes, desired_error)).start()

    def train_nn(self, nn, X_train, y_train, X_val, y_val, classes, desired_error):
        """Realiza el entrenamiento en un hilo separado."""
        try:
            best_accuracy = 0
            modelo_path = os.path.join(self.models_dir, "modelo_neural.pkl")
            epoch = 0
            loss = float('inf')
            start_time = time.time()
            while loss > desired_error:
                epoch += 1
                loss = nn.train_step(X_train, y_train)
                self.losses.append(loss)  # Guardar la pérdida
                # Evaluar en el conjunto de validación
                y_pred_val, _ = nn.predict(X_val)
                val_accuracy = self.calculate_accuracy(y_val, y_pred_val)
                if val_accuracy >= best_accuracy:
                    best_accuracy = val_accuracy
                    # Guardar el mejor modelo
                    nn.save_model(modelo_path)
                # Actualizar la salida cada 10 épocas
                if epoch % 10 == 0 or epoch == 1:
                    elapsed_time = time.time() - start_time
                    self.queue.put(('output', f"Época {epoch}, Pérdida: {loss:.6f}, Precisión Validación: {val_accuracy * 100:.3f}%, Tiempo: {elapsed_time:.2f}s\n"))
                    # Actualizar la gráfica
                    self.queue.put(('update_plot', None))
            # Indicar que se alcanzó el error deseado
            self.queue.put(('output', f"Entrenamiento completado en época {epoch}, Pérdida: {loss:.6f}\n"))
            # Mostrar resultados finales
            report = self.classification_report(y_val, y_pred_val, classes)
            self.queue.put(('output', f"Mejor precisión en validación: {best_accuracy * 100:.3f}%\n"))
            self.queue.put(('output', "Reporte de clasificación en Validación:\n"))
            self.queue.put(('output', report))
            self.queue.put(('status', f"Estado: Entrenamiento completado. Mejor precisión en validación: {best_accuracy * 100:.3f}%"))
            self.queue.put(('progress_stop', None))
            self.queue.put(('update_plot', None))
            self.queue.put(('messagebox', ("Entrenamiento", f"Entrenamiento completado.\nMejor precisión en validación: {best_accuracy * 100:.3f}%\nModelo guardado en:\n{modelo_path}")))
        except Exception as e:
            self.queue.put(('error', f"Ocurrió un error durante el entrenamiento:\n{e}"))
            self.queue.put(('status', "Estado: Error durante el entrenamiento."))
            print(f"Error durante el entrenamiento: {e}")

    def process_queue(self):
        """Procesa los mensajes en la cola para actualizar la interfaz gráfica."""
        try:
            while True:
                message_type, value = self.queue.get_nowait()
                if message_type == 'progress':
                    self.progress['value'] = value
                    self.update_idletasks()
                elif message_type == 'progress_stop':
                    self.progress.stop()
                elif message_type == 'output':
                    self.text_output.insert(tk.END, value)
                    self.text_output.see(tk.END)
                elif message_type == 'status':
                    self.status_label.config(text=value)
                elif message_type == 'messagebox':
                    title, message = value
                    messagebox.showinfo(title, message)
                elif message_type == 'error':
                    messagebox.showerror("Error en Entrenamiento", value)
                    self.status_label.config(text="Estado: Error durante el entrenamiento.")
                elif message_type == 'update_plot':
                    self.update_plot()
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

    def update_plot(self):
        """Actualiza la gráfica de error vs. épocas."""
        self.ax.clear()
        self.ax.plot(range(1, len(self.losses) + 1), self.losses, label='Error de Entrenamiento')
        self.ax.set_title("Error vs. Épocas")
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Error")
        self.ax.legend()
        self.canvas_fig.draw()

    def calculate_accuracy(self, y_true, y_pred):
        """Calcula la precisión del modelo."""
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / total if total > 0 else 0

    def classification_report(self, y_true, y_pred, classes):
        """Genera un reporte de clasificación básico."""
        report_text = ""
        for idx, cls in enumerate(classes):
            true_positive = np.sum((y_true == idx) & (y_pred == idx))
            false_positive = np.sum((y_true != idx) & (y_pred == idx))
            false_negative = np.sum((y_true == idx) & (y_pred != idx))
            support = np.sum(y_true == idx)

            precision = true_positive / (true_positive + false_positive + 1e-15)
            recall = true_positive / (true_positive + false_negative + 1e-15)
            f1_score = 2 * precision * recall / (precision + recall + 1e-15)

            report_text += f"Clase: {cls}\n"
            report_text += f"  Precisión: {precision:.4f}\n"
            report_text += f"  Recall: {recall:.4f}\n"
            report_text += f"  F1-Score: {f1_score:.4f}\n"
            report_text += f"  Soporte: {support}\n\n"

        return report_text