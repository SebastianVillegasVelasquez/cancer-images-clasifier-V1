from model.DataPreparation import DataPreparation
from model.sub_models.EfficientNetV2 import EfficientNetV2Instance
from utils.TaskType import  TaskType
from model.DataPreparation import DataPreparation
if __name__ == '__main__':

    pipeline = DataPreparation(
        root_folder='data/dataset_binary',
        img_size=(224, 224),
        color_mode='rgb',
        batch_size=32,
        class_mode='binary',
        shuffle=True)
    # model = EfficientNetV2Instance(
    #     layer_name='block6a_expand_activation',
    #     train_dataset=datasets[0],
    #     test_dataset=datasets[1],
    #     validation_dataset=datasets[2],
    #     task_type=TaskType.BINARY.value
    # )
    #
    # model.build_model()
    # model.execute_model_flow()