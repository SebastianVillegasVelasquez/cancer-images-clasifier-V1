import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

activations = [
    tf.keras.layers.ReLU(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.ELU(alpha=1.0),
    tf.keras.activations.sigmoid,
    tf.keras.activations.tanh,
    tf.keras.activations.softplus,
    tf.keras.activations.swish,
    tf.keras.activations.gelu
]

def test_model_with_different_activations(train_data, test_data, img_shape=(64, 64, 3), learning_rate=0.001, epochs=10):
    results = {}

    for activation in activations:
        # Get activation name for display
        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__
        else:
            activation_name = activation.__name__

        print(f"\nTesting with activation function: {activation_name}")

        # Create layers list for the model
        layers = [tf.keras.layers.Input(shape=img_shape)]

        # Add first Conv2D block
        layers.append(tf.keras.layers.Conv2D(16, (3, 3), padding='same',
                            kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        layers.append(tf.keras.layers.BatchNormalization())

        # Add activation - handle both layer and function types
        if hasattr(activation, '__class__') and 'keras.layers' in str(activation.__class__):
            # It's already a layer, add it directly
            layers.append(activation)
        else:
            # It's a function, wrap it in an Activation layer
            layers.append(tf.keras.layers.Activation(activation))

        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

        # Add second Conv2D block
        layers.append(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        layers.append(tf.keras.layers.BatchNormalization())

        # Add activation - handle both layer and function types
        if hasattr(activation, '__class__') and 'keras.layers' in str(activation.__class__):
            # Clone the layer for reuse
            if isinstance(activation, tf.keras.layers.ReLU):
                layers.append(tf.keras.layers.ReLU())
            elif isinstance(activation, tf.keras.layers.LeakyReLU):
                layers.append(tf.keras.layers.LeakyReLU(alpha=activation.alpha))
            elif isinstance(activation, tf.keras.layers.ELU):
                layers.append(tf.keras.layers.ELU(alpha=activation.alpha))
            else:
                # Fallback for other layer types
                layers.append(tf.keras.layers.Activation(activation))
        else:
            # It's a function, wrap it in an Activation layer
            layers.append(tf.keras.layers.Activation(activation))

        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

        # Add third Conv2D block
        layers.append(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=tf.keras.regularizers.L2(0.001)))
        layers.append(tf.keras.layers.BatchNormalization())

        # Add activation - handle both layer and function types
        if hasattr(activation, '__class__') and 'keras.layers' in str(activation.__class__):
            # Clone the layer for reuse
            if isinstance(activation, tf.keras.layers.ReLU):
                layers.append(tf.keras.layers.ReLU())
            elif isinstance(activation, tf.keras.layers.LeakyReLU):
                layers.append(tf.keras.layers.LeakyReLU(alpha=activation.alpha))
            elif isinstance(activation, tf.keras.layers.ELU):
                layers.append(tf.keras.layers.ELU(alpha=activation.alpha))
            else:
                # Fallback for other layer types
                layers.append(tf.keras.layers.Activation(activation))
        else:
            # It's a function, wrap it in an Activation layer
            layers.append(tf.keras.layers.Activation(activation))

        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

        # Add final layers
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        layers.append(tf.keras.layers.Dense(256))

        # Add activation - handle both layer and function types
        if hasattr(activation, '__class__') and 'keras.layers' in str(activation.__class__):
            # Clone the layer for reuse
            if isinstance(activation, tf.keras.layers.ReLU):
                layers.append(tf.keras.layers.ReLU())
            elif isinstance(activation, tf.keras.layers.LeakyReLU):
                layers.append(tf.keras.layers.LeakyReLU(alpha=activation.alpha))
            elif isinstance(activation, tf.keras.layers.ELU):
                layers.append(tf.keras.layers.ELU(alpha=activation.alpha))
            else:
                # Fallback for other layer types
                layers.append(tf.keras.layers.Activation(activation))
        else:
            # It's a function, wrap it in an Activation layer
            layers.append(tf.keras.layers.Activation(activation))

        layers.append(tf.keras.layers.Dropout(0.5))

        # Output layer always uses sigmoid for binary classification
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Create the model with the layers list
        model = tf.keras.models.Sequential(layers)

        # Use the same learning rate for all activation functions
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Train the model
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            verbose=1
        )

        # Evaluate the model
        evaluation = model.evaluate(test_data)

        # Store results - use the activation name we determined earlier
        results[activation_name] = {
            'history': history.history,
            'evaluation': evaluation
        }

        print(f"Evaluation with {activation_name}: Loss = {evaluation[0]:.4f}, Accuracy = {evaluation[1]:.4f}, AUC = {evaluation[2]:.4f}")

    return results


if __name__ == "__main__":
    # Example usage of the functions
    import tensorflow as tf

    # Define image properties
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    IMG_SHAPE = (64, 64, 3)

    # Create data generators
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    # Load data if available
    try:
        # Try to load training data
        train_data = image_generator.flow_from_directory(
            '../data/train',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            color_mode='rgb'
        )

        # Try to load test data
        test_data = image_generator.flow_from_directory(
            '../data/test',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            color_mode='rgb'
        )

        print("Data loaded successfully. Testing different activation functions...")

        # Test different activation functions
        results = test_model_with_different_activations(
            train_data=train_data,
            test_data=test_data,
            img_shape=IMG_SHAPE,
            learning_rate=0.001,
            epochs=5  # Using fewer epochs for demonstration
        )

        # Plot the results
        plot_activation_results(results)

        # Print summary of best performing activation function
        best_activation = max(results.items(), key=lambda x: x[1]['evaluation'][1])
        print(f"\nBest performing activation function based on accuracy: {best_activation[0]}")
        print(f"Accuracy: {best_activation[1]['evaluation'][1]:.4f}")
        print(f"AUC: {best_activation[1]['evaluation'][2]:.4f}")

    except Exception as e:
        print(f"Error loading data or running tests: {e}")
        print("Please make sure the data directories exist and contain the required structure.")
        print("Expected structure:")
        print("  data/")
        print("    train/")
        print("      class1/")
        print("      class2/")
        print("    test/")
        print("      class1/")
        print("      class2/")
