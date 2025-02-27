import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, l2_lambda=0.001):
        # Инициализация параметров с улучшенной инициализацией
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.l2_lambda = l2_lambda

        # Инициализация оптимизатора
        self.adam = AdamOptimizer(
            params=[self.W1, self.b1, self.W2, self.b2],
            lr=0.001,
            beta1=0.9,
            beta2=0.999
        )

    def forward(self, X):
        # Прямое распространение
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return x > 0

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        # Расчет потерь с L2-регуляризацией
        m = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        l2_loss = 0.5 * self.l2_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return cross_entropy + l2_loss

    def backward(self, X, y_true):
        # Обратное распространение
        m = X.shape[0]

        # Градиенты выходного слоя
        dZ2 = self.a2 - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dZ2) + self.l2_lambda * self.W2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Градиенты скрытого слоя
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_deriv(self.z1)
        dW1 = (1/m) * np.dot(X.T, dZ1) + self.l2_lambda * self.W1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Обновление параметров
        self.adam.step([dW1, db1, dW2, db2])

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            # Обновление моментов
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)

            # Коррекция смещения
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Обновление параметров
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def to_one_hot(y, num_classes=10):
    # Конвертация в one-hot encoding
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y] = 1.0
    return one_hot