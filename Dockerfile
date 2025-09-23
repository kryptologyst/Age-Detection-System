# Modern Age Detection System - Docker Configuration

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p sample_images

# Expose ports for Streamlit and Gradio
EXPOSE 8501 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV GRADIO_SERVER_PORT=7860

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "streamlit" ]; then\n\
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "gradio" ]; then\n\
    python gradio_app.py\n\
else\n\
    echo "Usage: docker run <image> [streamlit|gradio]"\n\
    echo "Default: streamlit"\n\
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh", "streamlit"]
