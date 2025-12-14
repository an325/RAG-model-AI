FROM python:3.11-slim

# Install tools
RUN apt-get update && apt-get install -y curl tar xz-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

EXPOSE 7860

# OLLAMA_HOST env pass hoga docker run se
CMD ["python", "app.py"]

