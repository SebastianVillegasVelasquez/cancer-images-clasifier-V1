from tensorflow import keras

def test_model(test_dataset):
    first_model = keras.models.load_model('model/saved_models/first_fit.keras')
    second_model = keras.models.load_model('model/saved_models/second_fit.keras')

    evaluation_first_model = first_model.evaluate(test_dataset)
    evaluation_second_model = second_model.evaluate(test_dataset)

    print("\n📊 Resultados Primer Modelo:")
    print(f"  {evaluation_first_model[0]:<10}: {evaluation_first_model[1]:.4f}")

    print("\n📊 Resultados Segundo Modelo:")
    print(f"  {evaluation_second_model[0]:<10}: {evaluation_second_model[1]:.4f}")