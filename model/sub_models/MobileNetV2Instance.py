from tensorflow import keras

from model.BaseModel import BaseModel


class MobileNetV2Instance(BaseModel):

    def __init__(self, *args, **kwargs):
        kwargs['loss'] = kwargs.get('loss', 'binary_crossentropy')
        super().__init__(*args, **kwargs)

    def build_model(self, model=None):
        if model is None:
            base_model = keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        else:
            base_model = model
        super().build_model(base_model=base_model)


    # super().set_callbacks(
    #     filepath_to_save_model='model/saved_models/mobile_net_v2.keras',
    #     tensorboard_log_file='logs_dir/binary_model'
    # )
