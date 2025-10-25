from typing import Union, Optional

from tensorflow import keras

from utils.preprocessing import image_augmentation


class BaseModel:

    def __init__(self,
                 model: Optional[keras.Model] = None,
                 optimizer: Optional[Union[str, keras.optimizers.Optimizer]] = None,
                 loss: Optional[Union[str, keras.losses.Loss]] = None,
                 metrics: Optional[Union[list[Union[str, keras.metrics.Metric]]]] = None,
                 epochs: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 train_dataset: Optional[object] = None,
                 test_dataset: Optional[object] = None,
                 validation_dataset: Optional[object] = None,
                 task_type: Optional[str] = None,
                 debug: Optional[bool] = None,
                 img_shape: Optional[tuple] = False,
                 layer_name: Optional[str] = None,
                 ):

        self.model = model
        self.optimizer = optimizer or keras.optimizers.Adam(learning_rate=0.001)
        self.metrics = metrics or ['accuracy']
        self.loss = loss
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = 0.01
        self.task_type = task_type
        self.debug = False
        self.layer_name = layer_name
        self.units = None
        self.epochs = epochs or 20

        match self.task_type:
            case 'binary':
                self.units = 1
            case 'categorical':
                self.units = 3

    def build_model(self, base_model):
        if base_model is None:
            raise ValueError("A base model must be provided.")
        # Removed index calculation from here
        print(f'[debug] building model with trainable base set to False')
        self.top_model(base_model=base_model)
        # Return None or remove return as index is not calculated here anymore
        return None

    def get_layer_name_index_to_fine_tuning_at(self, model):
        # This method is now primarily used internally by fine_tuning_model
        layers = [layer.name for layer in model.layers]
        if self.layer_name and self.layer_name in layers:
             return layers.index(self.layer_name)
        elif self.layer_name:
            print(f"[Warning] Layer name '{self.layer_name}' not found in model layers. Returning -1.")
            return -1 # Or handle error appropriately
        else:
             print("[Warning] layer_name is not set. Returning -1 for fine-tuning index.")
             return -1


    def fine_tuning_model(self, start_layer: Union[int, str]):
        """
        Unfreezes layers in the model starting from the specified index or layer name for fine-tuning.

        This method sets the `trainable` attribute of layers to True from the given
        `start_layer` (index or name) onwards. BatchNormalization layers are kept frozen.

        :param start_layer: The index (int) or name (str) of the layer from which
                            to start fine-tuning.
                            If index is negative, it counts from the end of the layers.
                            Layers with an index less than the determined starting index
                            will be frozen.
        """
        if self.model is None:
            raise RuntimeError("Model is not built yet. Call build_model() first.")

        actual_index = -1 # Default to no fine-tuning

        if isinstance(start_layer, str):
            # Find the index if a layer name is provided
            layers = [layer.name for layer in self.model.layers]
            try:
                actual_index = layers.index(start_layer)
                print(f"[debug] Fine-tuning from layer '{start_layer}' at index: {actual_index}")
            except ValueError:
                print(f"[Error] Layer with name '{start_layer}' not found in the model. Skipping fine-tuning.")
                return # Exit if layer name not found
        elif isinstance(start_layer, int):
            # Use the provided index (handle negative indices)
            if start_layer < 0:
                actual_index = len(self.model.layers) + start_layer
                if actual_index < 0:
                    print(f"[Warning] Negative index {start_layer} is out of bounds. Will not unfreeze any layers.")
                    return # Exit if index is out of bounds
                print(f"[debug] fine-tuning from layer index: {actual_index} (calculated from negative index {start_layer})")
            else:
                actual_index = start_layer
                if actual_index >= len(self.model.layers):
                    print(f"[Warning] Index {start_layer} is out of bounds. Will not unfreeze any layers.")
                    return # Exit if index is out of bounds
                print(f"[debug] fine-tuning from layer index: {actual_index}")
        elif start_layer is None:
             print("[debug] start_layer is None. Skipping fine-tuning.")
             return # No fine-tuning if start_layer is None
        else:
            print(f"[Error] Invalid type for start_layer: {type(start_layer)}. Must be int, str, or None.")
            return # Exit for invalid type


        # Apply trainability based on actual_index
        if actual_index != -1: # Only proceed if a valid index was determined
            for i, layer in enumerate(self.model.layers):
                if i >= actual_index:
                    # Keep BatchNormalization layers frozen during fine-tuning
                    if not isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = True
                    else:
                         layer.trainable = False # Explicitly set BatchNormalization to False
            else:
                layer.trainable = False # Ensure layers before the index are frozen


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
        outputs = keras.layers.Dense(self.units, activation='sigmoid')(x)
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
            metrics=self.metrics
        )

    def fit(self):

        if not hasattr(self, "callbacks") or not self.callbacks:
            raise RuntimeError("Callbacks must be set before training. Call set_callbacks() first.")

        if self.validation_dataset is not None:
            return self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=self.epochs,
                validation_data=self.validation_dataset,
                callbacks=self.callbacks
            )
        else:
            return self.model.fit(
                self.train_dataset,
                batch_size=32,
                epochs=self.epochs,
                callbacks=self.callbacks
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

    def show_layer_names(self, model=None):

        if model is None:
            print("[Error] model parameter can not be None")
            return
        # Use self.model if no model is passed
        model_to_show = model if model is not None else self.model
        if model_to_show is None:
             print("[Error] Model is not built yet.")
             return

        for idx, layer in enumerate(model_to_show.layers):
            print(f"{idx}: {layer.name}")


    def execute_model_flow(self):
        """
            Execute the complete training workflow for transfer learning.
            The workflow consists of:
                1. Building the base transfer learning model.
                2. Compiling the model with the default optimizer.
                3. Performing initial training (feature extraction stage).
                4. Unfreezing selected layers for fine-tuning.
                5. Recompiling the model with a lower learning rate (1e-5).
                6. Training again (fine-tuning stage).

            :return: None
            :rtype: None
            """
        # Build model sets base_model trainable to False initially
        self.build_model() # Assuming subclasses will provide base_model if needed

        self.compile()
        print("\nStarting initial training (feature extraction)...")
        self.fit()

        # Directly pass self.layer_name to fine_tuning_model
        # fine_tuning_model will handle finding the index if it's a string
        start_layer_for_fine_tuning = self.layer_name if self.layer_name else None # Use layer_name if set

        if start_layer_for_fine_tuning: # Proceed with fine-tuning only if a starting layer is specified
            print(f"\nStarting fine-tuning from: {start_layer_for_fine_tuning}...")
            # Now call the modified fine_tuning_model with the determined starting layer (name or index)
            self.fine_tuning_model(start_layer=start_layer_for_fine_tuning)
            # Only recompile and fit if fine-tuning was successfully initiated
            if any(layer.trainable for layer in self.model.layers): # Check if any layers were unfrozen
                 self.compile(learning_rate=1e-5) # Recompile with lower learning rate for fine-tuning
                 self.fit()
            else:
                 print("\nNo layers were unfrozen for fine-tuning. Skipping fine-tuning training stage.")

        else:
            print("\nSkipping fine-tuning as no starting layer name was specified in the constructor.")
