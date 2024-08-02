from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import requests
from PIL import Image
import io
import torch
from torchvision import transforms
import clip
from urllib.parse import urlencode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
PUBLIC_KEY = 'https://disk.yandex.ru/d/Oo-2DDTVd9p_nA'
MODEL_PATH = 'clip_cb_model.cbm'


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
download_model(PUBLIC_KEY, MODEL_PATH)
logger.info(f'{MODEL_PATH} downloaded')

logger.info(f'Loading {MODEL_PATH}')
clip_cb_model = CatBoostClassifier()
clip_cb_model.load_model(MODEL_PATH)
logger.info(f'{MODEL_PATH} loaded')

logger.info('Loading CLIP model')
clip_model, _ = clip.load("ViT-B/32", device="cpu")
logger.info('CLIP model loaded')


def clip_preprocess_image(image):
    logger.debug('Preprocessing image')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    transformed_image = transform(image)
    return transformed_image


@app.route('/predict', methods=['POST'])
def predict_image():
    logger.info('Received prediction request')
    try:
        predictions_list = []
        files = request.files.getlist('image')
        logger.info(f'Processing {len(files)} files')

        for file in files:
            allowed_extensions = {'jpg', 'jpeg'}
            if file.filename.split('.')[-1].lower() not in allowed_extensions:
                logger.warning(f'File {file.filename} has unsupported extension')
                continue

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            transformed_image = clip_preprocess_image(img).unsqueeze(0)

            with torch.no_grad():
                image_features = clip_model.encode_image(transformed_image).cpu().numpy()

            predictions = clip_cb_model.predict_proba(image_features)
            spam = bool(predictions[0][1] > 0.5)
            predictions_list.append({'confidence': float(predictions[0][1]), 'spam': bool(spam)})

        logger.info('Prediction completed')
        return jsonify(predictions_list)
    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info('Starting CLIPCB application')
    app.run(debug=True, host='0.0.0.0', port=5001)
