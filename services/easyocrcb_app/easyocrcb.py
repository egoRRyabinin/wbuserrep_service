from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from PIL import Image
import easyocr
import pickle
import io

app = Flask(__name__)

vectorizer = pickle.load(open('tfidf_vectorizer', 'rb'))
easyocr_cb_model = CatBoostClassifier()
easyocr_cb_model.load_model('easyocr_cb_model')
reader = easyocr.Reader(['ru', 'en'], gpu=True)


@app.route('/predict', methods=['POST'])
def predict_image():
    predictions_list = []
    files = request.files.getlist('image')

    for file in files:
        allowed_extensions = {'jpg', 'jpeg'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            continue  # Пропускаем файлы с неподдерживаемым форматом

        # Загружаем изображение и предобрабатываем его
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        extract_text = reader.readtext(
            img, paragraph=False,
            decoder='greedy',
            rotation_info=[0, 90, 270]
        )

        text_part = []

        for i in extract_text:
            _, text, _ = i
            if len(text) > 2:
                text_part.append(text)
            else:
                pass

        text = [" ".join(text_part)]
        vectorized_text = vectorizer.transform(text)

        predictions = easyocr_cb_model.predict_proba(vectorized_text)
        spam = bool(predictions[0][1] > 0.5)
        predictions_list.append({'confidence': float(predictions[0][1]), 'spam': bool(spam)})

    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
