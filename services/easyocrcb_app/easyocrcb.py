from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from PIL import Image
import requests
import easyocr
import pickle
import io
from urllib.parse import urlencode
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
MODEL_PUBLIC_KEY = "https://disk.yandex.ru/d/n-6KhR_r4qOQng"
MODEL_PATH = "easyocr_cb_model.cbm"
VECTORIZER_PUBLIC_KEY = "https://disk.yandex.ru/d/_MweSWzq4AE5BA"
VECTORIZER_PATH = "tfidf_vectorizer"


def download_model(public_key, save_path):
    try:
        final_url = BASE_URL + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']
        logger.info(f'Download URL obtained')
        download_response = requests.get(download_url)

        with open(save_path, 'wb') as f:
            f.write(download_response.content)
        logger.info(f'Model downloaded and saved')
    except requests.RequestException as e:
        logger.error(f'Failed to download model: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise

logger.info('Starting model download')
download_model(MODEL_PUBLIC_KEY, MODEL_PATH)
download_model(VECTORIZER_PUBLIC_KEY, VECTORIZER_PATH)
logger.info(f'{MODEL_PATH, VECTORIZER_PATH} downloaded')

logger.info(f'Loading {MODEL_PATH}')
easyocr_cb_model = CatBoostClassifier()
easyocr_cb_model.load_model(MODEL_PATH)
logger.info(f'{MODEL_PATH} loaded')

logger.info(f'Loading {VECTORIZER_PATH}')
vectorizer = pickle.load(open('tfidf_vectorizer', 'rb'))
logger.info(f'{VECTORIZER_PATH} loaded')

logger.info('Loading EasyOCR Reader')
reader = easyocr.Reader(['ru', 'en'], gpu=True)
logger.info('EasyOCR Reader loaded')


@app.route('/predict', methods=['POST'])
def predict_image():
    logger.info('Received prediction request')
    predictions_list = []
    files = request.files.getlist('image')

    for file in files:
        allowed_extensions = {'jpg', 'jpeg'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            logger.warning(f'File {file.filename} has unsupported extension and will be skipped')
            continue  # Пропускаем файлы с неподдерживаемым форматом

        try:
            # Загружаем изображение и предобрабатываем его
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            logger.info(f'Processing image {file.filename}')

            extract_text = reader.readtext(
                img, paragraph=False,
                decoder='greedy',
                rotation_info=[0, 90, 270]
            )
            logger.info(f'Text extracted from image {file.filename}')

            text_part = []
            for i in extract_text:
                _, text, _ = i
                if len(text) > 2:
                    text_part.append(text)

            text = [" ".join(text_part)]
            vectorized_text = vectorizer.transform(text)
            predictions = easyocr_cb_model.predict_proba(vectorized_text)
            spam = bool(predictions[0][1] > 0.5)
            predictions_list.append({'confidence': float(predictions[0][1]), 'spam': bool(spam)})
            logger.info(f'Prediction for image {file.filename}: {predictions[0][1]}')
        except Exception as e:
            logger.error(f'Error processing image {file.filename}: {e}')
            return jsonify({'error': f'Failed to process image {file.filename}'}), 500

    logger.info('Prediction request completed')
    return jsonify(predictions_list)


if __name__ == '__main__':
    logger.info('Starting EASYOCR application')
    app.run(debug=True, host='0.0.0.0', port=5002)
