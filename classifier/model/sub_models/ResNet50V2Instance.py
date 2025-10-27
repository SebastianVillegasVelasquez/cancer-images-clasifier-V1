from model.BaseModel import BaseModel
from tensorflow import keras

class ResNet50V2Instance(BaseModel):

    def __init__(self, *args, **kwargs):
        kwargs['loss'] = kwargs.get('loss', 'binary_crossentropy')
        super().__init__(*args, **kwargs)

        super().set_callbacks(
            filepath_to_save_model='model/saved_models/ResNet50V2Instance.keras',
            tensorboard_log_file='logs_dir/categorical_model'
        )

    def build_model(self, model=None):
        if model is None:
            base_model = keras.applications.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        else:
            base_model = model
        super().build_model(base_model=base_model)

    def fine_tuning_model(self):
        model = self.model
        fine_tuning_at = -50
        for layer in model.layers[:fine_tuning_at]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        self.model = model