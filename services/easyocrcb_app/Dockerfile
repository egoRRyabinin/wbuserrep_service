FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn torch torchvision easyocr pillow catboost scikit-learn

WORKDIR /app

COPY ["easyocr_cb_model", "tfidf_vectorizer", "easyocrcb.py", "./"]

EXPOSE 5002