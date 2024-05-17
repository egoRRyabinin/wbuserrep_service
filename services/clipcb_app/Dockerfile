FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn torch torchvision openai-clip pillow catboost

WORKDIR /app

COPY ["clip_cb_model", "clipcb.py", "./"]

EXPOSE 5001