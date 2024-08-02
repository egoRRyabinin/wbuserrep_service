FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn catboost requests

WORKDIR /app

COPY ["metamodel.py", "./"]

EXPOSE 5003