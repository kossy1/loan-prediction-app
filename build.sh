#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install numpy==1.24.3 --no-cache-dir
pip install pandas==2.0.3 --no-cache-dir
pip install scikit-learn==1.3.2 --no-cache-dir
pip install xgboost==1.7.6 --no-cache-dir
pip install joblib==1.3.2 --no-cache-dir
pip install fastapi==0.109.0 --no-cache-dir
pip install uvicorn==0.27.0 --no-cache-dir
pip install pydantic==2.5.3 --no-cache-dir
pip install starlette==0.35.1 --no-cache-dir
pip install streamlit==1.29.0 --no-cache-dir
pip install python-dotenv==1.0.0 --no-cache-dir
pip install python-multipart==0.0.6 --no-cache-dir
pip install httpx==0.26.0 --no-cache-dir

echo "Build completed successfully!"