FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnotify4 \
    libsdl2-2.0-0 \
    libwebkit2gtk-4.0-37 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models
RUN python scripts/download_models.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Run application
CMD ["python", "src/main.py"]