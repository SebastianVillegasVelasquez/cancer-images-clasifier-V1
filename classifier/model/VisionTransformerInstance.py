from model.BaseModel import BaseModel


class VisionTransformerInstance(BaseModel):

    def __init__(self, *args, **kwargs):
        kwargs['loss'] = kwargs.get('loss', 'binary_crossentropy')
        super().__init__(*args, **kwargs)

    def build_model(self, base_model=None):
        if base_model is None:
            base_model = ViTImageClassifier(
                num_classes=3,  # O 1 si es binario
                input_shape=(224, 224, 3),
                include_rescaling=True,
                include_top=False,
                pretrained="imagenet"
            )
        super().build_model(base_model=base_model)
