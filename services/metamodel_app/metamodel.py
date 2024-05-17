from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
from PIL import Image
import easyocr
import pickle
import io

app = Flask(__name__)

blended_model = CatBoostClassifier()
blended_model.load_model('blended_model')


def get_predictions(url, files):
    response = requests.post(url, files=files)
    return response

@app.route('/predict', methods=['POST'])
def predict_image():

    uploaded_files = request.files.getlist('image')
    files = [("image", (file.filename, file.stream, file.mimetype)) for file in uploaded_files]
    response1 = get_predictions("http://clipcb_app:5001/predict", files)
    # response2 = get_predictions("http://easyocr_app:5002/predict", files)
    # response1 = requests.post("http://clipcb_app:5001/predict", files=files)
    # response2 = requests.post("http://easyocr_app:5002/predict", files=files)

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
        predictions_list.append(bool(spam))

    predicted_values = []
    for i in range(len(response1.json())):
        resp1 = response1.json()
        # resp2 = response2.json()
        print("---------> resp1", resp1)
        # print("---------> resp2", resp2)
        model1_pred = resp1[i]['confidence']
        model2_pred = predictions_list[i]
        # model2_pred = resp2[i]['spam']
        # predicted_values.append([model1_pred, model2_pred])
        predicted_values.append([model1_pred, model2_pred])

    for prediction in predicted_values:
        metamodel_pred = blended_model.predict(prediction)
        spam = bool(metamodel_pred[0][0] > 0.5)
        predictions_list.append({'confidence': float(metamodel_pred[0][1]), 'spam': spam})

    return jsonify(predictions_list)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
