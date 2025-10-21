from model.DataPreparation import DataPreparation
from model.sub_models.EfficientNetV2 import EfficientNetV2Instance
from model.sub_models.MobileNetV2Instance import MobileNetV2Instance
from model.sub_models.ResNet50V2Instance import ResNet50V2Instance


def train_models_and_evaluate_assembly():
    models = get_list_models()
    datasets = get_datasets()
    for model in models:
        model.train_dataset = datasets[0]
        model.test_dataset = datasets[1]
        model.validation_dataset = datasets[2]
        model.build_model()
        model.execute_model_flow()


def get_list_models():
    models = [EfficientNetV2Instance(),
              MobileNetV2Instance(),
              ResNet50V2Instance()]



def get_datasets():
    return (DataPreparation(
        root_folder='data',
        img_size=224,
        batch_size=32,
        color_mode='rgb',
        class_mode='binary'
    ).create_datasets_from_subdirectories())
