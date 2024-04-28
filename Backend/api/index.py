from flask import Flask
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from supabase import create_client, Client

url: str = "https://whqnperemsymnmpfsoyi.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndocW5wZXJlbXN5bW5tcGZzb3lpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTQyNjIzNTksImV4cCI6MjAyOTgzODM1OX0.hU8VUWDVuX6DEP9QhoC8hOjmXlOJUuLcho7Guqxp_G4"

supabase: Client = create_client(url, key)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict')
def predict():
    # Download latest pic from supabase
    response = supabase.table('data').select("*").execute()

    # image = preprocessImage(file)

    # prediction = CNN(image)

    # Update DB with Prediction

    return response


def preprocessImage(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, (28, 28))
        image = image.astype(np.float32) / 255.0
    return image

def CNN(image):
    model = keras.Sequential([
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