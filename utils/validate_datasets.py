import os

def validate_datasets()-> bool:
    data_folder_path = '../data'
    return os.path.exists(data_folder_path)