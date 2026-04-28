import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = tf.keras.models.load_model("fruit_model.h5")

class_names = [
    'Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana',
    'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine',
    'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes',
    'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach',
    'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum',
    'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato',
    'Watermelon'
]

test_folder = r"testing dataset path "

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    if os.path.isfile(img_path):
        img = load_img(img_path, target_size=(150, 150))
        imgarray = img_to_array(img)
        imgarray = np.expand_dims(imgarray, axis=0)

        prediction = model.predict(imgarray, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"{img_name} --> {predicted_class} ({confidence:.2f}%)")