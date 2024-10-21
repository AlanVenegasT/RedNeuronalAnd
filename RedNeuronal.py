import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # Entradas para la función AND
y = np.array([[0], [0], [0], [1]], dtype=float)  # Salidas esperadas (función AND)

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.hiddenSize = 4  # Aumentamos el número de neuronas ocultas
        self.outputSize = 1

        # Inicialización de los pesos
        self.W1 = np.random.uniform(low=-0.1, high=0.1, size=(self.inputSize, self.hiddenSize))
        self.W2 = np.random.uniform(low=-0.1, high=0.1, size=(self.hiddenSize, self.outputSize))

    def forward(self, X):
        # Cálculo de activaciones de las neuronas
        self.z1 = np.dot(X, self.W1)  # Producto punto de las entradas con los pesos W1
        self.a1 = self.relu(self.z1)  # Aplicar función ReLU en la capa oculta
        self.z2 = np.dot(self.a1, self.W2)  # Producto punto de la capa oculta con los pesos W2
        a2 = self.sigmoid(self.z2)  # Salida final después de la sigmoide
        return a2

    def relu(self, z):
        # Función de activación ReLU
        return np.maximum(0, z)

    def reluPrime(self, z):
        # Derivada de la función ReLU
        return np.where(z > 0, 1, 0)

    def sigmoid(self, z):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Derivada de la función sigmoide
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def backward(self, X, y, a2):
        # Propagación hacia atrás
        self.a2_error = y - a2  # Error de salida
        self.a2_delta = self.a2_error * self.sigmoidPrime(self.z2)  # Aplicar derivada de la sigmoide al error de salida

        self.a1_error = self.a2_delta.dot(self.W2.T)  # Error de la capa oculta
        self.a1_delta = self.a1_error * self.reluPrime(self.z1)  # Aplicar derivada de ReLU al error de la capa oculta

        # Ajuste de los pesos
        learning_rate = 0.1  # Mantengo la tasa de aprendizaje
        self.W1 += learning_rate * X.T.dot(self.a1_delta)
        self.W2 += learning_rate * self.a1.T.dot(self.a2_delta)

    def train(self, X, y):
        a2 = self.forward(X)
        self.backward(X, y, a2)

    def saveWeights(self):
        # Guardar los pesos
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print("Predicciones después de entrenamiento: ")
        print("Entrada: " + str(X))
        print("Salida real: " + str(self.forward(X)))

# Crear una instancia de la clase Neural_Network
NN = Neural_Network()

# Número de iteraciones
for i in range(100000):  # Número alto de iteraciones para garantizar la convergencia
    print("Iteración: " + str(i))
    print("Entrada: \n" + str(X))
    print("Salida deseada: \n" + str(y))
    print("Salida real: \n" + str(NN.forward(X)))
    print("Error:" + str(np.mean(np.square(y - NN.forward(X)))))  # Mantener los prints tal como los necesitas
    print("\n")
    NN.train(X, y)

# Realizar predicciones
NN.predict()
