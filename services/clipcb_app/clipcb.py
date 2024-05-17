from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from PIL import Image
import io
import torch
from torchvision import transforms
import clip

app = Flask(__name__)

clip_cb_model = CatBoostClassifier()
clip_cb_model.load_model('clip_cb_model')
clip_model, _ = clip.load("ViT-B/32", device="cpu")


def clip_preprocess_image(image):
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
    predictions_list = []
    files = request.files.getlist('image')

    for file in files:
        allowed_extensions = {'jpg', 'jpeg'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            continue

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        transformed_image = clip_preprocess_image(img).unsqueeze(0)
        with torch.no_grad():
            image_features = clip_model.encode_image(transformed_image).cpu().numpy()

        predictions = clip_cb_model.predict_proba(image_features)
        spam = bool(predictions[0][1] > 0.5)
        predictions_list.append({'confidence': float(predictions[0][1]), 'spam': bool(spam)})
    print(predictions_list)

    return jsonify(predictions_list)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
