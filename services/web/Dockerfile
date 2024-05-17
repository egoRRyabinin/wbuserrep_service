FROM python:3.9.19-slim

RUN python -m pip install flask flask-cors gunicorn requests

WORKDIR /app

COPY ["app.py", "./"]
COPY templates/ templates/

EXPOSE 5000

#CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "app:app" ]