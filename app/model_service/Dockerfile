# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию, которая будет содержать наш пакет model_service
WORKDIR /app_root 

# Сначала копируем только requirements.txt для кэширования слоев Docker
COPY requirements.txt /app_root/model_service/requirements.txt

# Устанавливаем рабочую директорию внутрь будущей папки пакета для установки зависимостей
WORKDIR /app_root/model_service
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org

# Возвращаемся в /app_root и копируем весь код сервиса модели
WORKDIR /app_root
COPY . /app_root/model_service 
# Теперь у нас структура /app_root/model_service/main.py и т.д.
# И /app_root/model_service/saved_model/

# Устанавливаем переменную окружения для уровня логирования
ENV LOG_LEVEL="INFO"
# Python будет автоматически искать модули в текущей рабочей директории /app_root
# и сможет найти пакет model_service

# Указываем порт, который будет слушать приложение
EXPOSE 8017

# Команда для запуска FastAPI приложения через Uvicorn
# Uvicorn будет запущен из /app_root и сможет найти model_service.main
CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8017", "--workers", "1"]