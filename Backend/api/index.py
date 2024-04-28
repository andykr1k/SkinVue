from flask import Flask
from cv2 import imread, IMREAD_GRAYSCALE, resize
import numpy as np
from tensorflow.keras import layers, Sequential

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict/<path>')
def predict(path):
    url = "https://whqnperemsymnmpfsoyi.supabase.co/storage/v1/object/public/pictures/"

    image_path = url + path

    # image = preprocessImage(file)

    # prediction = CNN(image)

    # Update DB with Prediction

    return image_path


def preprocessImage(image):
    image = imread(image, IMREAD_GRAYSCALE)
    if image is not None:
        image = resize(image, (28, 28))
        image = image.astype(np.float32) / 255.0
    return image

def CNN(image):
    model = Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(40, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(7, activation='softmax')
    ])
    model.load_weights('../weights/train_1.h5')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    image = image.reshape(1, 28, 28, 1)
    predictions = model.predict(image)
    arr = np.round(predictions)
    print(arr)
    return arr