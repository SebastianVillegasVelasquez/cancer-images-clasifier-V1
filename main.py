from models.models import BenignMalignClassifier, MalignClassifier
import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':
    IMG_SIZE = (224, 224)

    # benign_malign_model = BenignMalignClassifier(
    #     train_path='./data/',
    #     test_path='./data/',
    #     img_size=IMG_SIZE,
    #     batch_size=32,
    #     model_save_path='./savedModel/benign_malign',
    #     channels=3
    # )


    #benign_malign_model.build_model()
    #benign_malign_model.train(epochs=50)

    # test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1 = benign_malign_model.model.evaluate(
    #     benign_malign_model.test_data, verbose=1
    # )

    print("\n=== Training Malignant Types Classifier ===")
    malign_model = MalignClassifier(
        train_path='./malign_cancer/train',
        test_path='./malign_cancer/test',
        img_size=IMG_SIZE,
        batch_size=32,
        model_save_path='./savedModel/malign_types',
        channels=3
    )

    malign_model.build_model()
    malign_model.train(epochs=50)

