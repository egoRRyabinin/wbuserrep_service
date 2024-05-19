from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import requests
from urllib.parse import urlencode
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

temp_dir = tempfile.TemporaryDirectory()
IMAGE_SAVE_PATH = temp_dir.name

BASE_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
MODEL_PUBLIC_KEY = "https://disk.yandex.ru/d/n-6KhR_r4qOQng"
MODEL_PATH = "easyocr_cb_model.cbm"
VECTORIZER_PUBLIC_KEY = "https://disk.yandex.ru/d/_MweSWzq4AE5BA"
VECTORIZER_PATH = "tfidf_vectorizer"
METAMODEL_PUBLIC_KEY = "https://disk.yandex.ru/d/5wzF96BR8FYfww"
METAMODEL_PATH = "metamodel.cbm"
# IMAGE_SAVE_PATH = "uploaded_images"

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)


def download_model(public_key, save_path):
    try:
        final_url = BASE_URL + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']
        logger.info(f'Download URL obtained')
        download_response = requests.get(download_url)

        with open(save_path, 'wb') as f:
            f.write(download_response.content)
        logger.info(f'Model {save_path} downloaded and saved')
    except requests.RequestException as e:
        logger.error(f'Failed to download model: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise


download_model(METAMODEL_PUBLIC_KEY, METAMODEL_PATH)

metamodel = CatBoostClassifier()
metamodel.load_model(METAMODEL_PATH)
logger.info(f'Model {METAMODEL_PATH} loaded')


def get_predictions(url, files):
    try:
        logger.info(f'Sending prediction request to {url}')
        response = requests.post(url, files=files)
        response.raise_for_status()
        logger.info(f'Received response from {url}')
        return response
    except requests.RequestException as e:
        logger.error(f'Failed to get predictions from {url}: {e}')
        raise


@app.route('/predict', methods=['POST'])
def predict_image():
    logger.info('Received prediction request')
    uploaded_files = request.files.getlist('image')
    saved_files = []

    for file in uploaded_files:
        filename = file.filename
        file_path = os.path.join(IMAGE_SAVE_PATH, filename)
        file.save(file_path)
        saved_files.append(file_path)
        logger.info(f'Saved {filename} to {file_path}')

    files1 = [("image", (os.path.basename(file), open(file, 'rb'), 'image/jpeg')) for file in saved_files]
    files2 = [("image", (os.path.basename(file), open(file, 'rb'), 'image/jpeg')) for file in saved_files]

    try:
        response1 = get_predictions("http://clipcb_app:5001/predict", files1)
    except Exception as e:
        logger.error(f'Error getting predictions from model 1: {e}')
        return jsonify({'error': 'Failed to process images with model 1'}), 500

    try:
        response2 = get_predictions("http://easyocr_app:5002/predict", files2)
    except Exception as e:
        logger.error(f'Error getting predictions from model 2: {e}')
        return jsonify({'error': 'Failed to process images with model 2'}), 500

    predicted_values = []
    resp1 = response1.json()
    resp2 = response2.json()
    for i in range(len(resp1)):
        model1_pred = resp1[i]['confidence']
        model2_pred = int(resp2[i]['spam'])
        predicted_values.append([model1_pred, model2_pred])

    final_predictions = []
    for prediction in predicted_values:
        try:
            metamodel_pred = metamodel.predict_proba(prediction)
            spam = bool(metamodel_pred[1] > 0.5)
            final_predictions.append({'confidence': float(metamodel_pred[1]), 'spam': spam})
        except Exception as e:
            logger.error(f'Error predicting with metamodel: {e}')
            return jsonify({'error': 'Failed to process metamodel predictions'}), 500

    logger.info('Prediction request completed')
    return jsonify(final_predictions)


if __name__ == '__main__':
    logger.info('Starting Flask application')
    app.run(debug=True, host='0.0.0.0', port=5003)
