# src/data_loader.py

import os
import json
import numpy as np
from PIL import Image, ImageEnhance

class DataLoader:
    def __init__(self, imagenes_guardadas_json_ruta, image_size=(64, 64), augment_data=True):
        self.imagenes_guardadas_json_ruta = imagenes_guardadas_json_ruta
        self.image_size = image_size
        self.mean = None
        self.std = None
        self.augment_data = augment_data

    def load_data(self):
        """Carga los datos y etiquetas desde el archivo JSON y aplica data augmentation si está habilitado."""
        if not os.path.exists(self.imagenes_guardadas_json_ruta):
            raise FileNotFoundError(f"No se encontró el archivo JSON en la ruta especificada:\n{self.imagenes_guardadas_json_ruta}")

        with open(self.imagenes_guardadas_json_ruta, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            raise ValueError("El archivo JSON está vacío.")

        inputs = []
        labels = []
        clases = set()

        # Identificar todas las clases
        for item in data:
            clases.add(item['tipo_pez'])

        clases = sorted(list(clases))  # Ordenar para consistencia
        clase_a_indice = {clase: idx for idx, clase in enumerate(clases)}
        print(f"Clases encontradas: {clase_a_indice}")

        for item in data:
            nombre_imagen = item['name']  # 'name' contiene el nombre del archivo de imagen
            tipo_pez = item['tipo_pez']
            ruta_imagen = os.path.join(os.path.dirname(self.imagenes_guardadas_json_ruta), nombre_imagen)

            if not os.path.exists(ruta_imagen):
                print(f"Advertencia: La imagen {ruta_imagen} no existe. Se omitirá.")
                continue

            try:
                imagen = Image.open(ruta_imagen).convert("RGB")
                imagen = imagen.resize(self.image_size, Image.LANCZOS)
                imagen_array = np.array(imagen) / 255.0  # Normalizar a [0, 1]
                input_flat = imagen_array.flatten()
                inputs.append(input_flat)
                labels.append(clase_a_indice[tipo_pez])

                if self.augment_data:
                    # Aplicar Data Augmentation
                    augmented_images = self.augment_image(imagen)
                    for aug_imagen in augmented_images:
                        imagen_array = np.array(aug_imagen) / 255.0
                        input_flat = imagen_array.flatten()
                        inputs.append(input_flat)
                        labels.append(clase_a_indice[tipo_pez])

                print(f"Imagen cargada y procesada: {ruta_imagen}, Clase: {tipo_pez}")
            except Exception as e:
                print(f"Error al procesar la imagen {ruta_imagen}: {e}")

        inputs = np.array(inputs)
        labels = np.array(labels)

        # Calcular la media y desviación estándar para normalización
        self.mean = np.mean(inputs, axis=0)
        self.std = np.std(inputs, axis=0) + 1e-8  # Evitar división por cero

        print(f"Total de imágenes cargadas: {len(inputs)}")
        return inputs, labels, clases

    def augment_image(self, image):
        """Genera variaciones de la imagen para aumentar el conjunto de datos."""
        augmented_images = []

        # Rotaciones
        for angle in [90, 180, 270]:
            rotated = image.rotate(angle)
            augmented_images.append(rotated)

        # Volteo horizontal
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped)

        # Cambios de brillo
        enhancer = ImageEnhance.Brightness(image)
        for factor in [0.7, 1.3]:
            bright = enhancer.enhance(factor)
            augmented_images.append(bright)

        # Cambios de contraste
        enhancer = ImageEnhance.Contrast(image)
        for factor in [0.7, 1.3]:
            contrast = enhancer.enhance(factor)
            augmented_images.append(contrast)

        return augmented_images

    def load_single_image(self, image_pil):
        """Procesa una sola imagen PIL y la prepara para la predicción."""
        try:
            imagen = image_pil.convert("RGB")
            imagen = imagen.resize(self.image_size, Image.LANCZOS)
            imagen_array = np.array(imagen) / 255.0  # Normalizar a [0, 1]
            input_flat = imagen_array.flatten()
            # Normalizar con la media y desviación estándar del entrenamiento
            if self.mean is not None and self.std is not None:
                input_flat = (input_flat - self.mean) / self.std
            else:
                print("Advertencia: La media y desviación estándar no están definidas.")
            return input_flat
        except Exception as e:
            print(f"Error al procesar la imagen para predicción: {e}")
            return None
