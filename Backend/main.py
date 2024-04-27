from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)

def preprocessImage(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, (28,28))
        image = image.astype(np.float32) / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file attached'
    if 'id' not in request.form:
        return 'No id attached'

    file = request.files['file']
    id = request.form['id']

    if file.filename == '':
        return 'No file present'
    if id == '':
        return 'No id present'

    image = preprocessImage(file)
    prediction = CNN(image)

    # Update DB with Prediction

    return 0


if __name__ == '__main__':
    app.run(debug=True)