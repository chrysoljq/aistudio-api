# Use Debian 12 (bookworm) explicitly so package names stay stable.
FROM python:3.14-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for Camoufox and Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core libraries for Firefox/Camoufox
    libgtk-3-0 \
    libglib2.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxtst6 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgdk-pixbuf2.0-0 \
    libnspr4 \
    libnss3 \
    libdrm2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libwayland-client0 \
    libwayland-egl1 \
    libwayland-server0 \
    # Fonts
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-wqy-zenhei \
    # Utilities
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY main.py .

# Create necessary directories
RUN mkdir -p /app/data /tmp

# Set permissions
RUN chmod +x /app/main.py

# Expose ports
# 8080: API server
# 9222: Camoufox debug port
EXPOSE 8080 9222

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/models || exit 1

# Default command
CMD ["python3", "main.py", "server", "--port", "8080", "--camoufox-port", "9222"]
