import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("dataset/clean_reviews.csv")

X = data["Clean Review"].astype(str)
y = data["Recommended IND"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

# ==========================
# Tokenizer
# ==========================

tokenizer = Tokenizer(

    num_words=10000,
    oov_token="<OOV>"

)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(

    X_train_seq,
    maxlen=150,
    padding="post",
    truncating="post"

)

X_test_pad = pad_sequences(

    X_test_seq,
    maxlen=150,
    padding="post",
    truncating="post"

)

# ==========================
# Build Model
# ==========================

model = Sequential()

model.add(
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=150
    )
)

model.add(
    Bidirectional(
        LSTM(64, return_sequences=False)
    )
)

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

model.summary()

# ==========================
# Compile
# ==========================

model.compile(

    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]

)

# ==========================
# Class Weights
# ==========================

weights = compute_class_weight(

    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train

)

class_weights = {

    0: weights[0],
    1: weights[1]

}

print(class_weights)

# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(

    monitor="val_loss",
    patience=3,
    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "models/lstm_model.keras",
    monitor="val_accuracy",
    save_best_only=True

)

# ==========================
# Train
# ==========================

history = model.fit(

    X_train_pad,
    y_train,

    epochs=10,

    batch_size=32,

    validation_data=(X_test_pad, y_test),

    callbacks=[early_stop, checkpoint],

    class_weight=class_weights

)

# ==========================
# Evaluate
# ==========================

loss, accuracy = model.evaluate(X_test_pad, y_test)

print()

print("Accuracy :", accuracy)
print("Loss :", loss)

# ==========================
# Save Tokenizer
# ==========================

with open("models/lstm_tokenizer.pkl", "wb") as file:

    pickle.dump(tokenizer, file)

print("Tokenizer Saved!")