FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PYTHONPATH=/app

COPY classifying_app/requirements.txt .

RUN pip install -r requirements.txt

COPY classifying_app /app/classifying_app

EXPOSE 8080

CMD ["python", "-u", "-m", "uvicorn", "classifying_app.api.server:app", "--host", "0.0.0.0", "--port", "8080"]