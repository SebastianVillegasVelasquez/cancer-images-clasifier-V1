from tensorflow import keras

from model.BaseModel import BaseModel


class MobileNetV2Instance(BaseModel):

    def __init__(self, *args, **kwargs):
        kwargs['loss'] = kwargs.get('loss', 'binary_crossentropy')
        kwargs['layer_name'] = kwargs.get('layer_name', 'block_14_expand')
        super().__init__(*args, **kwargs)

        super().set_callbacks(
            filepath_to_save_model='model/saved_models/MobileNetV2Instance.keras',
            tensorboard_log_file='logs_dir/categorical_model'
        )

    def build_model(self, base_model=None):
        if base_model is None:
            base_model = keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )

            self.show_layer_names(model=base_model)
        else:
            base_model = base_model
        super().build_model(base_model=base_model)