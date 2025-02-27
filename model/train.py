import os

import numpy as np
from tensorflow.keras.datasets import mnist
from neural_net import NeuralNetwork, to_one_hot

def load_data():
    # Загрузка и подготовка данных
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Нормализация и преобразование формы
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # Разделение на train/validation
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Загрузка данных
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Преобразование меток
    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)
    y_test_oh = to_one_hot(y_test)

    # Параметры модели
    input_size = 784
    hidden_size = 256
    output_size = 10
    l2_lambda = 0.0001

    # Инициализация модели
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        l2_lambda=l2_lambda
    )

    # Гиперпараметры
    epochs = 50
    batch_size = 128

    # Цикл обучения
    best_val_acc = 0.0
    best_params = None

    for epoch in range(1, epochs+1):
        # Обучение мини-батчами
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train_oh[i:i+batch_size]

            # Forward + Backward
            pred = model.forward(X_batch)
            model.backward(X_batch, y_batch)

        # Валидация
        val_pred = model.forward(x_val)
        val_loss = model.compute_loss(y_val_oh, val_pred)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch}/{epochs}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'W1': model.W1.copy(),
                'b1': model.b1.copy(),
                'W2': model.W2.copy(),
                'b2': model.b2.copy()
            }

    # Тестирование
    model.W1, model.b1, model.W2, model.b2 = best_params.values()
    test_pred = model.forward(x_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Сохраняем лучшие параметры (если они были найдены) в файл my_mnist_params.npz
    if best_params is not None:
        save_path = os.path.join(os.path.dirname(__file__), "my_mnist_params.npz")
        np.savez(save_path,
                 W1=best_params['W1'],
                 b1=best_params['b1'],
                 W2=best_params['W2'],
                 b2=best_params['b2'])
        print(f"Обученные параметры сохранены в: {save_path}")

    # --- Дополнительно: загрузка параметров из npz-файла ---
    # Например, если нужно сразу применить лучшие параметры
    loaded_params = np.load(save_path)
    model.W1 = loaded_params['W1']
    model.b1 = loaded_params['b1']
    model.W2 = loaded_params['W2']
    model.b2 = loaded_params['b2']

    # Тестирование
    test_pred = model.forward(x_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()