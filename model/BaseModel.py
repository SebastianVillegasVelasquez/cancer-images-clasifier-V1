from tensorflow import keras
from typing import Union, Optional
from utils.preprocessing import image_augmentation

class callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Perform custom actions here, e.g., save predictions, adjust learning rate, etc.
        if logs.get('val_accuracy') > 0.9:
            print("Validation accuracy reached 90%, stopping training.")
            self.model.stop_training = True

class ModelBenignMalignClassifier:
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

    def build_model(self, model = None):
        inputs = keras.Input(shape=(256, 256, 3))
        x = image_augmentation(inputs)
        x = keras.layers.Conv2D(32, (3, 3), padding='same', name='Conv1')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(32, (3, 3), padding='same', name='Conv2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        block_1_output = keras.layers.MaxPooling2D((2, 2), name='MaxPool1')(x)

        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='Conv3')(block_1_output)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='Conv4')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.MaxPooling2D((2, 2), name='MaxPool2')(x)

        shortcut = keras.layers.Conv2D(64, (1, 1), strides=2, padding="same")(block_1_output)
        shorcut = keras.layers.BatchNormalization()(shortcut)
        shorcut = keras.layers.ELU()(shortcut)
        block_2_output = keras.layers.add([x, shortcut])

        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='Conv5')(block_2_output)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='Conv6')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        shortcut = keras.layers.Conv2D(64, (1, 1), strides=2, padding="same")(block_2_output)
        block_3_output = keras.layers.add([x, shortcut])

        x = keras.layers.Conv2D(128, (3, 3), padding='same', name='Conv7')(block_3_output)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same', name='Conv8')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        shortcut = keras.layers.Conv2D(128, (1, 1), strides=2, padding="same")(block_3_output)
        block_4_output = keras.layers.add([x, shortcut])

        x = keras.layers.MaxPooling2D((2, 2))(block_4_output)
        x = keras.layers.Conv2D(128, (3, 3), padding='same', name='Conv9')(block_4_output)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)

        x = keras.layers.GlobalMaxPool2D()(block_4_output)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)

        model_resnet = keras.Model(inputs, output, name="ResNetLike")
        self.model = model
        return model_resnet

    def build_transfer_learning_model(self, base_model = None):
        base_model = keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(256, 256, 3)
        )
        #Feature extraction
        base_model.trainable = False
        inputs = keras.Input(shape=(256, 256, 3))
        encoder_output = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(encoder_output)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        self.model = model

        self.model.compile()

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
                validation_data=self.validation_dataset,
                callbacks=[self.callbacks]
            )
            model.save('./model/saved_models/first_fit.keras')
            self.number_iterarion += 1
        else:
            self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=20,
                validation_data=self.validation_dataset,
                callbacks=[self.callbacks]
            )
            model.save('./model/saved_models/second_fit.keras')

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
        #Second flow
        self.fine_tuning_model()
        self.compile(learning_rate=1e-5)
        self.fit()


