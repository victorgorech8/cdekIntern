FROM ollama/ollama:latest

# Устанавливаем Python
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаем виртуальное окружение для Python
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

EXPOSE 8000 11434

ENTRYPOINT ["sh", "-c", "ollama serve & sleep 15 && ollama pull llama3.2 && ollama pull nomic-embed-text && uvicorn app.main:app --host 0.0.0.0 --port 8000"]