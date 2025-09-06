# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies needed for the build (includes dev headers for pip packages)
RUN apt-get update && apt-get install -y \
    gcc g++ libffi-dev libssl-dev python3-dev pkg-config \
    libhdf5-dev libjpeg-dev libpng-dev libfreetype6-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final production image
FROM python:3.12-slim

WORKDIR /app

# Install only the system dependencies required at runtime
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p data/datasets data/models data/chunks data/database logs

# Expose port and set entrypoint
EXPOSE 8001
ENV PORT=8001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]