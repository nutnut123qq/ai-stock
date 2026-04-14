FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/vendor

COPY build/python/ ./vendor/
COPY main.py ./
COPY src/ ./src/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
