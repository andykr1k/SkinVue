# SkinVue

## Description

SkinVue is a wearable device and platform designed to scan skin lesions and classify them as benign or malignant. In cases of malignancy, it generates a distribution indicating the likelihood of specific diseases or cancers. Our wearable device ensures consistently high-quality images, which are transmitted to our server housing a convolutional neural network for classification. Following classification, the image and its categorization are stored in the user's account for historical reference and future tracking.

## Tech Stack

- Frontend: React, Tailwind, Framer Motion and Redux
- Server: Flask on AWS EC2 Instance
- Database: Supabase
- Hardware: Arduino Uno R3 and ArduCam Mini
- CNN Model: Python with Tensorflow
- 3D Model: Fusion 360

## Contributors

- Andrew Krikorian: Frontend, Backend, and AI Model
- Omar Nosseir: 3D Printing and Hardware
- Jacob Hensley: Hardware
- Osvaldo Ortiz: Hardware and Documentation