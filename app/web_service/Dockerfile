# web_service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем статические файлы и шаблоны
COPY ./static ./static
COPY ./templates ./templates

# Копируем исходный код веб-приложения и файл с опциями
COPY app.py .
COPY available_options.py . 

EXPOSE 5019

# Команда для запуска Flask приложения (пример с gunicorn для продакшена)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
# Для разработки можно использовать:
CMD ["python", "app.py"]