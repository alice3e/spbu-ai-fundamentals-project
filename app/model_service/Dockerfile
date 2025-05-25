# model_service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем папку с артефактами модели
COPY ./saved_model ./saved_model

# Копируем исходный код сервиса модели
COPY predictor.py .
COPY main.py .

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8017

# Команда для запуска FastAPI приложения с Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8017"]