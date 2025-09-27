from model.MobileNetV2Instance import MobileNetV2Instance

if __name__ == '__main__':
   model = MobileNetV2Instance()
   model.build_model()
   model.view_model_layers()
