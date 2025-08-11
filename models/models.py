import os

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.metrics import F1Score


class BaseModel:
    def __init__(self, train_path, test_path, img_size=(224, 224), batch_size=32, class_mode='binary',
                 model_save_path='./savedModel', channels=3):
        self.train_path = train_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.model_save_path = model_save_path
        self.use_smote = True
        self.channels = channels

        self.img_shape = (self.img_size[0], self.img_size[1], self.channels)

        os.makedirs(self.model_save_path, exist_ok=True)

        self._setup_data_generators()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        # Setup learning rate scheduler
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )

        self.metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                restore_best_weights=True,
                patience=15,
                min_delta=0.0005
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]

        self.model = None

    def _setup_data_generators(self):

        image_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.9, 1.1],  # More conservative brightness to preserve color features
            fill_mode='reflect',  # Reflect is better for medical images
            shear_range=0.1,  # Limited shear to preserve shape
            validation_split=0.2,
            # Contrast adjustment - important for dermatological features
            preprocessing_function=lambda x: tf.image.random_contrast(x, lower=0.9, upper=1.1)
        )

        # No augmentation for test data, just rescaling
        image_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )

        # Forzar color_mode a 'rgb' si channels=3
        color_mode = 'rgb' if self.channels == 3 else 'grayscale'
        print(f"[INFO] Usando color_mode='{color_mode}' para los generadores de datos.")
        self.train_data = image_generator_train.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            color_mode=color_mode,
        )

        self.test_data = image_generator_test.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            color_mode=color_mode,
            shuffle=False
        )

        batch_x, _ = next(self.train_data)
        if self.channels == 3 and batch_x.shape[-1] != 3:
            print(
                f"[ADVERTENCIA] Las imágenes cargadas tienen {batch_x.shape[-1]} canales, se esperaban 3. Verifica tus datos de entrada.")

        print("Class indices:", self.train_data.class_indices)
        print("Class distribution in training data:", dict(zip(
            self.train_data.class_indices.keys(),
            np.bincount(self.train_data.classes)
        )))

    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model()")

    def apply_smote(self, X, y):
        """Apply SMOTE to generate synthetic samples for minority classes"""
        print("Applying SMOTE to balance classes...")

        # Reshape the data for SMOTE (flatten images)
        n_samples = X.shape[0]
        flat_shape = (n_samples, np.prod(X.shape[1:]))
        X_flat = X.reshape(flat_shape)

        # Convert y to int for bincount
        y_int = y.astype(int)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_int)

        # Reshape back to original shape
        X_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0],) + X.shape[1:])

        print(f"Original class distribution: {np.bincount(y_int)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled

    def collect_training_data(self):
        train_images = []
        train_labels = []

        self.train_data.reset()

        batch_count = 0
        max_batches = len(self.train_data)

        while batch_count < max_batches:
            batch_images, batch_labels = next(self.train_data)
            train_images.append(batch_images)
            if self.class_mode == 'binary':
                train_labels.append(batch_labels.flatten())
            else:
                train_labels.append(batch_labels)
            batch_count += 1

        train_images = np.vstack(train_images)

        if self.class_mode == 'binary':
            train_labels = np.concatenate(train_labels)
        else:
            train_labels = np.vstack(train_labels)
            train_labels = np.argmax(train_labels, axis=1)

        return train_images, train_labels

    def train(self, epochs=50, class_weight=None, verbose=1):
        if self.model is None:
            self.build_model()

        if self.use_smote:
            X_train, y_train = self.collect_training_data()

            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)

            if self.class_mode == 'categorical':
                num_classes = len(np.unique(y_train_resampled))
                y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes)

            history = self.model.fit(
                X_train_resampled, y_train_resampled,
                validation_data=self.test_data,
                epochs=epochs,
                callbacks=self.callbacks,
                class_weight=class_weight,
                verbose=verbose
            )
        else:
            history = self.model.fit(
                self.train_data,
                validation_data=self.test_data,
                epochs=epochs,
                callbacks=self.callbacks,
                class_weight=class_weight,
                verbose=verbose
            )

        print(self.model.summary())

        self.showPlotsModel(history)
        return history

    def showPlotsModel(self, history):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def testModel(self, img, model):
        img = tf.keras.preprocessing.image.load_img(img, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        if self.class_mode == 'binary':
            return 'Malignant' if predictions[0][0] > 0.5 else 'Benign'
        elif self.class_mode == 'categorical':
            class_indices = self.train_data.class_indices
            class_labels = list(class_indices.keys())
            predicted_class = np.argmax(predictions, axis=1)[0]
            return class_labels[predicted_class]


class BenignMalignClassifier(BaseModel):
    def __init__(self, train_path, test_path, img_size=(224, 224), batch_size=32,
                 model_save_path='./savedModel/benign_malign', channels=3):
        super().__init__(train_path, test_path, img_size, batch_size, class_mode='binary',
                         model_save_path=model_save_path, channels=3)

    def build_model(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_shape
        )
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=self.metrics
        )

        return self.model

    def train(self, epochs=50, verbose=1):
        if self.model is None:
            self.build_model()

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_data.classes),
            y=self.train_data.classes
        )
        class_weights_dict = dict(enumerate(class_weights))

        print("Using class weights:", class_weights_dict)

        return super().train(epochs=epochs, class_weight=class_weights_dict, verbose=verbose)

    def testModel(self, img, model):
        img = tf.keras.preprocessing.image.load_img(img, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        if self.class_mode == 'binary':
            return 'Malignant' if predictions[0][0] > 0.5 else 'Benign'
        elif self.class_mode == 'categorical':
            class_indices = self.train_data.class_indices
            class_labels = list(class_indices.keys())
            predicted_class = np.argmax(predictions, axis=1)[0]
            return class_labels[predicted_class]


class MalignClassifier(BaseModel):
    def __init__(self, train_path, test_path, img_size=(224, 224), batch_size=32,
                 model_save_path='./savedModel/malign_types', channels=3):
        super().__init__(train_path, test_path, img_size, batch_size, class_mode='categorical',
                         model_save_path=model_save_path, channels=channels)

    def build_model(self):

        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_shape
        )

        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=self.metrics
        )

        return self.model

    def train(self, epochs=100, verbose=1):
        if self.model is None:
            self.build_model()

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_data.classes),
            y=self.train_data.classes
        )
        class_weights_dict = dict(enumerate(class_weights))

        print("Using class weights:", class_weights_dict)

        from sklearn.model_selection import KFold

        train_images = []
        train_labels = []

        batch_count = 0
        max_batches = len(self.train_data)

        self.train_data.reset()

        while batch_count < max_batches:
            batch_images, batch_labels = next(self.train_data)
            train_images.append(batch_images)
            train_labels.append(batch_labels)
            batch_count += 1

        train_images = np.vstack(train_images)
        train_labels = np.vstack(train_labels)

        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_no = 1
        val_scores = []

        for train_idx, val_idx in kf.split(train_images):
            print(f'Training fold {fold_no}/{k_folds}')

            x_train_fold, x_val_fold = train_images[train_idx], train_images[val_idx]
            y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]

            history = self.model.fit(
                x_train_fold, y_train_fold,
                validation_data=(x_val_fold, y_val_fold),
                epochs=epochs // k_folds,
                callbacks=self.callbacks,
                class_weight=class_weights_dict,
                verbose=verbose
            )
            val_score = self.model.evaluate(x_val_fold, y_val_fold, verbose=0)
            val_scores.append(val_score)

            print(f'Fold {fold_no} validation score: {val_score}')
            fold_no += 1

        avg_val_score = np.mean(val_scores, axis=0)
        print(f'Average validation score across {k_folds} folds: {avg_val_score}')

        print("Final training on all data")
        history = self.model.fit(
            self.train_data,
            validation_data=self.test_data,
            epochs=epochs // 2,
            callbacks=self.callbacks,
            class_weight=class_weights_dict,
            verbose=verbose
        )

        print(self.model.summary())
        self.showPlotsModel(history)

        return history