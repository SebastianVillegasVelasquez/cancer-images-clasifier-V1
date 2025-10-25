"""
Ejemplo sencillo de uso de Keras Tuner (keras_tuner) para tuning de hiperparámetros.
- Dataset: CIFAR-10 convertido a problema binario (clases 0-4 -> 0, 5-9 -> 1)
- Modelo: pequeña CNN configurable vía `hp` (número de capas convolucionales, filtros,
  tamaño de kernel, unidades dense, dropout, tasa de aprendizaje)
- Tuner: RandomSearch (rápido para demo)

Cómo adaptarlo a tu proyecto:
- Reemplaza la función `build_model` por un constructor que reciba `hp` y construya
  tu arquitectura (por ejemplo, MobileNetV2/ResNet50V2 con top personalizado).
- Usa tu `dataset['train']` y `dataset['validation']` (tf.data.Dataset) en lugar de
  los arrays numpy (puedes convertirlos con `tf.data.Dataset.unbatch()` o
  prediciendo sobre el dataset, pero para `tuner.search()` es más directo usar
  arrays numpy o un `tf.data.Dataset` bien preparado).

Ejecución rápida (demo):
- Se ejecutan pocas búsquedas (max_trials=6, epochs=3) para que sea rápido en CPU.
- Ajusta `max_trials` y `epochs` cuando corras en GPU o quieras explorar más.

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Importar el módulo de Keras Tuner (se instala como keras-tuner y se importa como keras_tuner)
try:
    import keras_tuner as kt
except Exception as e:
    raise RuntimeError("keras_tuner no está instalado. Ejecuta: pip install keras-tuner")


def load_cifar10_binary():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Convertir a etiqueta binaria: clases 0-4 -> 0 , 5-9 -> 1 (ejemplo didáctico)
    y_train = (y_train.flatten() >= 5).astype(np.int32)
    y_test = (y_test.flatten() >= 5).astype(np.int32)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # separar validación rápida desde train
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print("Shapes:", x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(hp):
    """Constructor del modelo que recibe un objeto `hp` de Keras Tuner.
    Devuelve un modelo compilado listo para entrenar.
    """
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs

    # Número de bloques convolucionales (1-3)
    conv_blocks = hp.Int('conv_blocks', 1, 3, default=2)

    for i in range(conv_blocks):
        filters = hp.Choice(f'filters_{i}', [32, 64, 128], default=64)
        kernel_size = hp.Choice(f'kernel_{i}', [3, 5], default=3)
        x = keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        if hp.Boolean(f'batchnorm_{i}', default=True):
            x = keras.layers.BatchNormalization()(x)
        if hp.Boolean(f'pool_{i}', default=True):
            x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)

    units = hp.Int('dense_units', 32, 256, step=32, default=128)
    x = keras.layers.Dense(units, activation='relu')(x)

    dropout = hp.Float('dropout', 0.0, 0.5, step=0.1, default=0.3)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_tuner_demo(max_trials=6, executions_per_trial=1):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_binary()

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='kt_dir',
        project_name='cifar_binary_demo',
        overwrite=True
    )

    print("Comenzando búsqueda de hiperparámetros con Keras Tuner")
    tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val), batch_size=64)

    print("Búsqueda finalizada. Mejores hiperparámetros:")
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    for hp_key in best_hp.values:
        print(f"  {hp_key}: {best_hp.get(hp_key)}")

    best_model = tuner.get_best_models(num_models=1)[0]
    print("Resumen del mejor modelo:")
    best_model.summary()

    print("Evaluando en test set:")
    test_res = best_model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_res[0]:.4f}, Test accuracy: {test_res[1]:.4f}")


if __name__ == '__main__':
    # Corrida rápida para demo; aumenta max_trials y epochs para búsquedas serias
    run_tuner_demo(max_trials=6, executions_per_trial=1)


# ===== Consejos para integrar con tu proyecto =====
# 1) Si quieres tunear una arquitectura basada en transfer learning (por ejemplo
#    MobileNetV2), escribe un `build_model(hp)` que cargue el backbone con
#    include_top=False y parametrice (a) si congelar capas, (b) dropout en la top
#    layer, (c) número de neuronas en la cabeza, (d) tasa de aprendizaje.
#
# 2) Para usar tus `tf.data.Dataset` preparados: convierte a arrays numpy con
#    `np.concatenate([x for x,_ in ds], axis=0)` y `np.concatenate([y for _,y in ds], axis=0)`
#    o usa `tuner.search(tf_dataset, epochs=..., validation_data=val_dataset)` si
#    tu dataset está bien configurado (algunos backends necesitan que sea repetible y batched).
#
# 3) Casos factibles que puedes tunear en tu proyecto:
#   - Tunar tasa de aprendizaje, batch size y número de neuronas en la cabeza.
#   - Tunar cuántas capas del backbone descongelar (fine-tuning point).
#   - Tunar la probabilidad de dropout y L2 regularization en la cabeza.
#   - Tunar opciones de data augmentation (p. ej. rotación, zoom ranges) y ver su efecto.
#   - Tunar arquitectura de la top model (número de dense layers, unidades) cuando uses transfer learning.
#
# 4) Cuando uses varios modelos (ensemble), puedes tunear cada sub-modelo
#    independientemente y luego evaluar combinaciones (por ejemplo, promediando
#    probabilidades) para optimizar la métrica final.
#
# 5) Guarda los mejores hiperparámetros con `tuner.get_best_hyperparameters()` y
#    reentrena una última vez con más épocas (`best_model.fit(...)`) antes de
#    guardar el modelo final.

