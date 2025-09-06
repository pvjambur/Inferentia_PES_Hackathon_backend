FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Create necessary directories
RUN mkdir -p data/datasets/text data/datasets/images data/datasets/audio \
    data/models data/chunks data/database logs

ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]