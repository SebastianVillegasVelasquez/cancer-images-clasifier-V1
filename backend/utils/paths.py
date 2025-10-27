from pathlib import Path

ROOT_PATH =  Path.cwd()

MODEL_PATH = ROOT_PATH.joinpath('model')

SAVED_KERAS_MODELS = MODEL_PATH.joinpath('saved_models')

SAVED_META_MODELS = MODEL_PATH.joinpath('meta_models')
