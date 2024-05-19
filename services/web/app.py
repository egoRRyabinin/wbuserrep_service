from flask import Flask, render_template, request, jsonify
import requests
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def index():
    logger.info('Rendering index page')
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        uploaded_files = request.files.getlist('image[]')
        selected_service = request.headers.get('selector')
        logger.info(f'Received {len(uploaded_files)} files for service {selected_service}')

        service_url = ""
        if selected_service == "clip_cb":
            service_url = "http://clipcb_app:5001/predict"
        elif selected_service == "easyocr_cb":
            service_url = "http://easyocr_app:5002/predict"
        elif selected_service == "metamodel":
            service_url = "http://metamodel_app:5003/predict"

        logger.info(f'Sending request to {selected_service}')
        files = [("image", (file.filename, file.stream, file.mimetype)) for file in uploaded_files]
        response = requests.post(service_url, files=files)

        if response.status_code == 200:
            logger.info('Successfully processed image(s)')
            return jsonify(response.json())
        else:
            logger.error(f'Failed to process image(s), status code: {response.status_code}')
            return jsonify({'error': 'Failed to process image'}), 500
    except Exception as e:
        logger.error(f'Error occurred: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info('Starting Flask application')
    app.run(debug=True, host='0.0.0.0', port=5000)
