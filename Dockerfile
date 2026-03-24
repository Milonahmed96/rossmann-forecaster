FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements WITHOUT torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch explicitly AFTER requirements
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall

COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY configs/ ./configs/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]