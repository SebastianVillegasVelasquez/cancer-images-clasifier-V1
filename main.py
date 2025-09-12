import keras.optimizers
import model.DataPreparation as model
from utils.validate_datasets import validate_datasets
from utils.create_dataset import create_data_set

if __name__ == '__main__':
    exists_datasets = validate_datasets()
    if not exists_datasets:

        create_data_set()
    else:
        data = model.DataPreparation('./data',
                                     (256, 256),
                                     'rgb',
                                     batch_size=32)

        train_dataset, test_dataset, validation_dataset = data.create_datasets_from_subdirectories()

        model = ModelBenignMalignClassifier(train_dataset=train_dataset,
                                            test_dataset=test_dataset,
                                            validation_dataset=validation_dataset)
        model.execute_model_flow(test_dataset)
