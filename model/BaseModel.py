from typing import Union, Optional

from tensorflow import keras

from utils.preprocessing import image_augmentation


class BaseModel:
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
                 train_dataset: Optional[object] = None,
                 test_dataset: Optional[object] = None,
                 validation_dataset: Optional[object] = None,
                 task_type: Optional[str] = None,
                 debug: Optional[bool] = None,
                 img_shape: Optional[tuple] = False
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
        self.metrics = metrics or ['accuracy']
        self.loss = loss
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = 0.001
        self.task_type = task_type
        self.debug = False
        if self.task_type == None:
            self.task_type = None
        elif self.task_type != None:
            self.loss = keras.losses.BinaryCrossentropy() \
                if self.task_type == 'binary' \
                else keras.losses.CategoricalCrossentropy()

    def build_model(self, base_model):
        base_model.trainable = False
        self.top_model(base_model=base_model)

    def top_model(self, base_model):
        inputs = keras.Input(
            shape=(224, 224, 3))
        x = image_augmentation(inputs)
        encoder_output = base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(encoder_output)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if self.task_type == "binary":
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = keras.layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        self.model = model

    def compile(self, learning_rate=None):
        """
           Compile the model with the specified optimizer, loss, and metrics.

           If a custom learning rate is provided, an Adam optimizer will be created
           with that learning rate. Otherwise, the default optimizer stored in
           ``self.optimizer`` will be used.

           :param learning_rate: Optional custom learning rate for the optimizer.
           :type learning_rate: float or None
           :return: None
           :rtype: None
           """
        if learning_rate is None:
            optimizer = self.optimizer
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

    def fit(self):
        """
            Train the model using the training dataset.

            This method handles one or multiple training iterations. On the first
            iteration, the model is saved as ``first_fit.keras``. On subsequent
            iterations, the model is saved as ``second_fit.keras``.

            :return: None
            :rtype: None
            """
        if not hasattr(self, "callbacks") or not self.callbacks:
            raise RuntimeError("Callbacks must be set before training. Call set_callbacks() first.")

        if self.validation_dataset is not None:
            return self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=20,
                validation_data=self.validation_dataset,
                callbacks=[self.callbacks]
            )
        else:
            return self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=20,
                callbacks=[self.callbacks]
            )

    def set_callbacks(self,
                      filepath_to_save_model: str,
                      tensorboard_log_file: str,
                      monitor_param: Optional[str] = 'val_loss',
                      callbacks: list[keras.callbacks.Callback] = None,
                      early_stopping_patiencie: int = 10,
                      reducelr_patiente: int = 5,
                      factor: float = 0.5,
                      ):
        if early_stopping_patiencie <= reducelr_patiente:
            raise ('Early Stopping patience callback must have to be greater than'
                   'ReduceLROnPlateau patience callback')

        self.callbacks = callbacks or [
            keras.callbacks.EarlyStopping(
                patience=early_stopping_patiencie,
                monitor=monitor_param,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=filepath_to_save_model,
                monitor=monitor_param,
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=reducelr_patiente,
                monitor=monitor_param,
                factor=factor
            ),
            keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_file,
                histogram_freq=1,
                write_images=True
            )
        ]

    def fine_tuning_model(self):
        """
            Enable fine-tuning of the model layers.

            This function unfreezes layers of the base model starting at a specific
            index (default 313), allowing them to be trainable, except for
            BatchNormalization layers which remain frozen. Updates ``self.model``
            with the new configuration.

            :return: None
            :rtype: None
            """
        model = self.model
        fine_tuning_at = 313
        for layer in model.layers[fine_tuning_at:]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        self.model = model

    def execute_model_flow(self):
        """
            Execute the complete training workflow for transfer learning.
            The workflow consists of:
                1. Building the base transfer learning model with DenseNet121.
                2. Compiling the model with the default optimizer.
                3. Performing initial training (feature extraction stage).
                4. Unfreezing selected layers for fine-tuning.
                5. Recompiling the model with a lower learning rate (1e-5).
                6. Training again (fine-tuning stage).

            :return: None
            :rtype: None
            """
        self.build_model()
        self.compile()
        self.fit()
        self.fine_tuning_model()
        self.compile(learning_rate=1e-5)
        self.fit()

    def view_model_layers(self):
        model = self.model
        layers = [layer for layer in model.layers]
        print(f'Number of layers:{len(layers)}')

        for layer in layers:
            print(layer)
