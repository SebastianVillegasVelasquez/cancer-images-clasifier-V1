import os
import numpy as np
import tensorflow as tf

def predict_image(image_path, model, img_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]
    predicted_label = "Maligno" if prediction >= 0.5 else "Benigno"

    true_label = os.path.basename(os.path.dirname(image_path))

    print(f"Imagen: {os.path.basename(image_path)}")
    print(f"Clase real: {true_label}")
    print(f"Predicción: {predicted_label} ({prediction:.2f})\n")

def predict_all_images(test_dir, model, img_size):
    for root, dirs, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                predict_image(img_path, model, img_size)

