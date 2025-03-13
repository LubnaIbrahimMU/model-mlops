# FROM python:3.9-slim
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app
 
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# RUN pip install --no-cache-dir torch>=1.9.0
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Ensure GPU is available and recognized
RUN python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
