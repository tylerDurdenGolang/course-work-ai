import os
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Update
from telegram.ext.callbackcontext import CallbackContext

import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()  # Загружаем переменные окружения из файла .env

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Шаг 1: Загружаем обученные параметры
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_mnist_params.npz")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}\n"
                            f"Сначала запустите обучение: python train.py")

data = np.load(MODEL_PATH)
params = {
    "W1": data["W1"],
    "b1": data["b1"],
    "W2": data["W2"],
    "b2": data["b2"]
}

def forward(X, params):
    """
    Прямое распространение для MLP:
    ReLU -> Softmax
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    # Слой 1 (ReLU)
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    # Выход (softmax)
    z2 = a1 @ W2 + b2
    expZ = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def start_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Привет! Я бот, распознающий рукописные цифры (0..9).\n"
        "Пришли мне фото или картинку цифры."
    )

def help_command(update: Update, context: CallbackContext):
    """ Обработка команды /help """
    update.message.reply_text(
        "📌 **Доступные команды:**\n"
        "/start - Начать работу с ботом.\n"
        "/help - Получить информацию о возможностях бота.\n\n"
        "📷 **Как использовать:**\n"
        "1. Отправь мне изображение с цифрой.\n"
        "2. Я распознаю её и пришлю тебе результат.\n"
        "3. Если цифра не распознана, попробуй отправить более четкое изображение."
    )

def handle_image(update: Update, context: CallbackContext):
    photo_file = update.message.photo[-1].get_file()
    file_path = "temp_digit.jpg"
    photo_file.download(file_path)

    # Открываем, приводим к 28x28, grayscale
    img = Image.open(file_path).convert('L').resize((28, 28))
    arr = np.array(img, dtype=np.float32)

    # 1) Инвертируем
    arr = 255 - arr

    # 2) (Опционально) Простая бинаризация
    #    Попробуйте использовать, если у вас шумный фон
    # arr[arr < 128] = 0
    # arr[arr >= 128] = 255

    arr = arr / 255.0
    arr = arr.reshape((1, -1))

    # pred = forward(arr, params)
    # digit = np.argmax(pred, axis=1)[0]

    # 4. Прогон через сеть
    pred = forward(arr, params)            # pred.shape -> (1, 10)
    pred = pred[0]                         # теперь (10, )
    digit = np.argmax(pred)                # индекс с максимальной вероятностью

    update.message.reply_text(f"Кажется, это цифра: {digit}")
    os.remove(file_path)

def main():
    # Загружаем токен из переменной окружения
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Переменная окружения TELEGRAM_BOT_TOKEN не установлена.")
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher

    # Команда /start
    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))
    # Обработка изображений
    dp.add_handler(MessageHandler(Filters.photo, handle_image))

    # Запускаем
    updater.start_polling()
    print("Бот запущен... Нажмите Ctrl+C, чтобы остановить.")
    updater.idle()

if __name__ == "__main__":
    main()
