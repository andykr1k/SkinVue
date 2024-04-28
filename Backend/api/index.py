from flask import Flask
from PIL import Image
import numpy as np
from keras import layers, Sequential

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
    with Image.open(image) as img:
        img_gray = img.convert('L')
        img_resized = img_gray.resize((28, 28))
        image_array = np.array(img_resized, dtype=np.float32) / 255.0
    return image_array

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