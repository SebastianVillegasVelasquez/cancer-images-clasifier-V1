from tensorflow import keras
from typing import Union, Optional
from utils.preprocessing import image_augmentation

class ModelBenignMalignClassifier:
    """
    This class represents a base model classifier,
    it has the basics for create, compile and fit
    the inherit models
    """
    def __init__(self,
                    model: Optional[keras.Model] = None,
                    optimizer: Optional[Union[str, keras.optimizers.Optimizer]] = None,
                    loss: Optional[Union[str, keras.losses.Loss]] = None,
                    metrics: Optional[Union[list[Union[str, keras.metrics.Metric]]]] = None,
                    learning_rate: Optional[float] = None,
                    train_dataset = None,
                    test_dataset = None,
                    validation_dataset = None
                    ):
        """
        Initialize the basic model class for managing model flow training.

        :param model: Keras model to be trained.
        :type model: Optional[keras.Model]

        :param optimizer: Optimizer to use (either a string with the optimizer name
                          or a keras.optimizers.Optimizer instance).
        :type optimizer: Optional[Union[str, keras.optimizers.Optimizer]]

        :param loss: Loss function (can be a string or a keras.losses.Loss instance).
        :type loss: Optional[Union[str, keras.losses.Loss]]

        :param metrics: List of metrics to monitor during training.
        :type metrics: Optional[list[Union[str, keras.metrics.Metric]]]

        :param learning_rate: Learning rate for the optimizer.
        :type learning_rate: Optional[float]

        :param train_dataset: Training dataset.
        :type train_dataset: Any

        :param test_dataset: Test dataset.
        :type test_dataset: Any

        :param validation_dataset: Validation dataset.
        :type validation_dataset: Any
        """
        self.model = model
        self.optimizer = optimizer or keras.optimizers.Adam(learning_rate=0.001)
        self.loss = loss or keras.losses.BinaryCrossentropy()
        self.metris = metrics or ['accuracy']
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = 0.001
        self.callbacks = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True

        )
        self.number_iterarion = 1

    def build_transfer_learning_model(self):
        """
         Build a transfer learning model using DenseNet121 as the base.

         This function initializes a DenseNet121 model pretrained on ImageNet
         for feature extraction, freezes its layers, and attaches a custom
         dense classifier block on top. The resulting model is stored in
         ``self.model``.

         :return: None. The model is assigned to ``self.model``.
         :rtype: None
         """

        base_model = keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(256, 256, 3)
        )
        #Feature extraction
        base_model.trainable = False
        inputs = keras.Input(shape=(256, 256, 3))
        x = image_augmentation(inputs)
        encoder_output = base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(encoder_output)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        self.model = model

    def compile(self, learning_rate=None):
        if learning_rate is None:
            optimizer = self.optimizer
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metris
        )

    def fit(self):
        if self.number_iterarion == 1:

            self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=20,
                validation_data=self.validation_dataset,
                callbacks=[self.callbacks]
            )
            self.model.save('./model/saved_models/first_fit.keras')
            self.number_iterarion += 1
        else:
            self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=20,
                validation_data=self.validation_dataset,
                callbacks=[self.callbacks]
            )
            self.model.save('./model/saved_models/second_fit.keras')

    def fine_tuning_model(self):
        model = self.model
        fine_tuning_at = 313
        print('Capas totales', len(model.layers))
        print('Capas entrenables', len(model.trainable_weights))
        for layer in model.layers[fine_tuning_at:]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        self.model = model
        print('Capas totales despues del fine tuning:', len(model.layers))
        print('Capas entrenables despues del fine tuning', len(model.trainable_weights))

    # build model(freezing all layers and including a classifier top)
    # -> compile
    # -> train
    # -> (unfreeze de last block of convolution and not the BN)
    # -> recompile
    # -> retrain for the last time
    def execute_model_flow(self):
        #first flow
        self.build_transfer_learning_model()
        self.compile()
        self.fit()
        #Second part of the flow flow
        self.fine_tuning_model()
        self.compile(learning_rate=1e-5)
        self.fit()


