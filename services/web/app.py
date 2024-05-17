from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_files = request.files.getlist('image[]')
    selected_service = request.headers.get('selector')
    service_url = ""
    # Отправляем изображения на выбранный сервис
    if selected_service == "service1":
        service_url = "http://clipcb_app:5001/predict"
    elif selected_service == "service2":
        service_url = "http://easyocr_app:5002/predict"
    elif selected_service == "service3":
        service_url = "http://metamodel_app:5003/predict"

    files = [("image", (file.filename, file.stream, file.mimetype)) for file in uploaded_files]
    # Отправляем запрос на сервис
    response = requests.post(service_url, files=files)

    if response.status_code == 200:
        # Если ответ успешный, возвращаем JSON ответа
        return jsonify(response.json())
    else:
        # Если произошла ошибка, возвращаем сообщение об ошибке
        return jsonify({'error': 'Failed to process image'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
