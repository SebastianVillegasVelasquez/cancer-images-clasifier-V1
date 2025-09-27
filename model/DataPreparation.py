import os.path
import tensorflow as tf
from utils.preprocessing import prepare_dataset


class DataPreparation:
    """
    This class makes semi-automatic the creation of datasets.
    """
    def __init__(self,
                 root_folder,
                 img_size,
                 color_mode,
                 batch_size,
                 class_mode:str = 'binary'):
        self.root_folder = root_folder
        self.img_size = img_size
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.class_mode = class_mode

    def create_datasets_from_subdirectories(self):

        subdirectories = os.listdir(self.root_folder)
        datasets = []
        for _ in subdirectories:
            path = os.path.join(self.root_folder, _)
            datasets.append(self.create_dataset(path))

        print('Data normalization done!')
        return datasets


    def create_dataset(self, path):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=path,
            image_size=self.img_size,
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            label_mode=self.class_mode
        )
        print('Dataset loaded successfully.')
        print('Classes found in dataset:', dataset.class_names)
        if 'train' in path:
            dataset = prepare_dataset(dataset, training=True)
        else:
            dataset = prepare_dataset(dataset)

        return dataset
