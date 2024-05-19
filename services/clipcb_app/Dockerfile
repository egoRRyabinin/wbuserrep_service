FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn torch torchvision openai-clip pillow catboost requests

WORKDIR /app

COPY ["clipcb.py", "./"]

EXPOSE 5001