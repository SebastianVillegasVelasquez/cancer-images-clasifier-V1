import keras.optimizers
import tensorflow as tf
import model.DataPreparation as model
from model.BaseModel import ModelBenignMalignClassifier
import numpy as np

if __name__ == '__main__':

    data = model.DataPreparation('./data',
                                 (256, 256),
                                 'rgb',
                                 batch_size=32)

    train_dataset, test_dataset, validation_dataset = data.create_datasets_from_subdirectories()

    model = ModelBenignMalignClassifier(train_dataset=train_dataset,
                                        test_dataset=test_dataset,
                                        validation_dataset=validation_dataset)
    model.execute_model_flow()

