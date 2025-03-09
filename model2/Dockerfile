# FROM python:3.9-slim
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


WORKDIR /app
 
COPY requirements.txt .


# RUN pip install --no-cache-dir torch>=1.9.0
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# uvicorn app:app --host 0.0.0.0 --port 8000
