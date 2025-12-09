FROM python:3.11-slim

# Установка зависимостей ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка PyTorch с CUDA 12.4 (GPU)
# Примечание: PyTorch 2.4.0+cu121 совместим с CUDA 12.4 благодаря обратной совместимости
# Если доступна версия с cu124, можно использовать её для оптимальной производительности
RUN pip install --no-cache-dir torch==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Установка остальных зависимостей
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    transformers==4.40.0 \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.5.0

# Копируем приложение
WORKDIR /app
COPY app.py .

# Порт
EXPOSE 8009

# Запуск
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8009"]