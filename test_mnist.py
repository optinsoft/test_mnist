import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from os import listdir, path
from pathlib import Path
import re
# import pytesseract
# from PIL import Image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the images
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Reshape the images and add a channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

def predict_digit(image_file_name: str, images_dir: str, prediction_dir: str):
    image_file_path = f'{images_dir}{image_file_name}'
    orig_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(orig_image, (28, 28))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    prediction = np.argmax(model.predict(image))
    print("Predicted Digit:", prediction)
    '''
    config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
    image_pil = Image.fromarray(orig_image)
    ocr = pytesseract.image_to_string(image_pil, config=config)
    print("Tesseract OCR:", ocr)
    '''
    # prediction_file_name = re.sub(r'\.([^.]+)$', f'.p{prediction}.\\1', image_file_name)
    prediction_file_name = f'p{prediction}.{image_file_name}'
    prediction_file_path = f'{prediction_dir}{prediction_file_name}'
    print(f"saving prediction => {prediction_file_path}")
    cv2.imwrite(prediction_file_path, orig_image)

images_dir = r'./img/digits/'
prediction_dir = r'./img/digits/prediction/'

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for image_file_name in listdir(images_dir):
    if path.isfile(f'{images_dir}{image_file_name}'):
        predict_digit(image_file_name, images_dir, prediction_dir)
