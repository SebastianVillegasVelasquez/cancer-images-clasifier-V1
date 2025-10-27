from typing import Dict

import glob
import joblib
import os

def load_all_joblib_models(dir_path: str) -> Dict[str, object]:
    models = {}
    pattern = os.path.join(dir_path, "*.joblib")
    for fpath in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(fpath))[0]
        model = load_joblib_model(fpath)
        if model is not None:
            models[name] = model
    return models

def load_joblib_model(filepath: str):
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None