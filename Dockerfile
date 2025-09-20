# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app and datasets (small datasets OK, otherwise fetch at runtime)
COPY app/ ./app
COPY datasets/ ./datasets

EXPOSE 5000
ENV MODEL_PATH=/app/datasets/student_model.pkl
CMD ["python", "app/server.py"]

