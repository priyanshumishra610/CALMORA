# Dockerfile for Calmora FastAPI Backend
FROM python:3.10-slim

WORKDIR /app

COPY ../configs/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy .env if present
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 