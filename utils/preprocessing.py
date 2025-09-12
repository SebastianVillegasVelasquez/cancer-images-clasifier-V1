from tensorflow import keras
import tensorflow as tf

def preprocess_data(x, y):
    """
    This function handles de image vector and rescale it and optimize it
    :param x: (image dataset) 
    :param y: (labels dataset)
    :return: x
    """
    output = tf.keras.layers.Rescaling(1. / 255)(x)
    return output, y

def prepare_dataset(dataset, training = False):
    dataset = dataset.map(lambda x, y: preprocess_data(x, y))
    dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(3100)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def image_augmentation(x):

    x = keras.layers.RandomFlip('horizontal')(x)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = keras.layers.RandomTranslation(0.2, 0.2)(x)
    x = keras.layers.RandomContrast(0.2)(x)
    return x


