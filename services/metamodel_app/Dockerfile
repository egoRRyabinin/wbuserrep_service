FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn torch torchvision easyocr pillow catboost scikit-learn requests

WORKDIR /app

COPY ["easyocr_cb_model", "tfidf_vectorizer", "blended_model", "metamodel.py", "./"]

EXPOSE 5003