FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn torch torchvision easyocr pillow catboost scikit-learn requests

WORKDIR /app

COPY ["easyocrcb.py", "./"]

EXPOSE 5002