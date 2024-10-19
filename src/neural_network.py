# src/neural_network.py

import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Inicializa la red neuronal con una capa oculta.

        Parámetros:
            - input_size: Número de neuronas en la capa de entrada.
            - hidden_size: Número de neuronas en la capa oculta.
            - output_size: Número de neuronas en la capa de salida.
            - learning_rate: Tasa de aprendizaje.
        """
        # Inicializar pesos y biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        """Función de activación ReLU."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivada de la función ReLU."""
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        """Función softmax para la capa de salida."""
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))  # Evitar overflow
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train_step(self, X, y):
        """
        Realiza un paso de entrenamiento utilizando retropropagación.

        Parámetros:
            - X: Datos de entrada.
            - y: Etiquetas verdaderas.
        """
        # Forward propagation
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.softmax(z2)

        # Convertir y a one-hot encoding
        y_onehot = np.zeros_like(a2)
        y_onehot[np.arange(len(y)), y] = 1

        # Calcular pérdida (cross-entropy)
        loss = -np.sum(y_onehot * np.log(a2 + 1e-15)) / len(y)

        # Backpropagation
        delta2 = a2 - y_onehot
        dW2 = a1.T.dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = delta2.dot(self.W2.T) * self.relu_derivative(z1)
        dW1 = X.T.dot(delta1)
        db1 = np.sum(delta1, axis=0)

        # Actualizar pesos y biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        return loss

    def predict(self, X):
        """
        Realiza una predicción sobre los datos de entrada X.

        Retorna:
            - predictions: Índices de las clases predichas.
            - confidences: Confianza asociada a cada predicción.
        """
        # Forward propagation
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.softmax(z2)

        predictions = np.argmax(a2, axis=1)
        confidences = np.max(a2, axis=1)
        return predictions, confidences

    def save_model(self, path):
        """Guarda el modelo entrenado en un archivo."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        """Carga un modelo entrenado desde un archivo."""
        with open(path, 'rb') as f:
            return pickle.load(f)
