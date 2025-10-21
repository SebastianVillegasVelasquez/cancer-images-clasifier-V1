from typing import Optional

from tensorflow import keras


from model.BaseModel import BaseModel


class EfficientNetV2Instance(BaseModel):
    def __init__(self, *args, **kwargs):
        self.model: Optional[keras.Model] = None
        kwargs['loss'] = kwargs.get('loss', 'binary_crossentropy')
        super().__init__(*args, **kwargs)

    def init_model(self, model=None):
        if model is None:
            self.model = keras.applications.EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        else:
            self.model = model

    def build_model(self, base_model=None):
        if base_model is None:
            base_model = keras.applications.EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        else:
            base_model = base_model
        super().build_model(base_model=base_model)
        super().set_callbacks(
            filepath_to_save_model
            ='model/saved_models/binary/efficient_net_v2_best_model.h5',
            tensorboard_log_file='logs_dir/binary_model'
        )

    def show_layer_names(self):
        super().show_layer_names(model=self.model)
