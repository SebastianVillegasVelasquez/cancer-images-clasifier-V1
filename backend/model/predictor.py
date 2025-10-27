from utils import utils
from utils.paths import  MODEL_PATH, SAVED_META_MODELS


def predict_image(image_bytes):
    pass



def get_models():
    meta_models = utils.load_all_joblib_models('model/meta_models')
    print(meta_models)

get_models()
