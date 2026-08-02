from tensorflow.keras.models import load_model

print("Loading old model...")

model = load_model(
    "models/lstm_model.keras",
    compile=False
)

print("Saving new model...")

model.save(
    "models/lstm_model_fixed.keras"
)

print("Done")