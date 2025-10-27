from model.DataPreparation import DataPreparation
from model.sub_models.EfficientNetV2 import EfficientNetV2Instance
from model.sub_models.MobileNetV2Instance import MobileNetV2Instance
from utils.TaskType import  TaskType
from model.DataPreparation import DataPreparation
if __name__ == '__main__':

    dataset = DataPreparation(
        root_folder='data',
        img_size=(224, 224),
        color_mode='rgb',
        batch_size=32,
        class_mode='binary',
        shuffle=True).create_datasets_from_subdirectories()

    mobilenet_model = MobileNetV2Instance(
        train_dataset=dataset['train'],  # Use the training dataset from DataPreparation
        validation_dataset=dataset['validation'],  # Use the validation dataset from DataPreparation
        test_dataset=dataset['test'],  # Use the test dataset from DataPreparation
        task_type=TaskType.BINARY.value,  # Or 'categorical' depending on your problem
        layer_name='block_15_expand',
        epochs=5
    )

    mobilenet_model.execute_model_flow()