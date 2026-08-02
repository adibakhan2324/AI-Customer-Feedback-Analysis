from tensorflow.keras.models import load_model

model = load_model(
    "models/lstm_model.keras",
    compile=False
)

model.save(
    "models/lstm_model.h5"
)

print("LSTM model converted successfully")