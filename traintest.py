import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from preprocessing import data_generator, SEQUENCE_LENGTH
import json
import os

# Model Configuration
with open("mapping.json") as f:
    MAPPING = json.load(f)
    
VOCAB_SIZE = len(MAPPING)  # Directly from loaded mapping
NUM_UNITS = [256, 256]
DROPOUT_RATE = 0.3
L2_REG = 0.001
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 128

def build_model():
    model = tf.keras.Sequential([
        layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
            input_length=SEQUENCE_LENGTH,
            name="embedding"
        ),
        layers.LSTM(
            NUM_UNITS[0],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(L2_REG),
            name="lstm_1"
        ),
        layers.Dropout(DROPOUT_RATE, name="dropout_1"),
        layers.LSTM(
            NUM_UNITS[1],
            kernel_regularizer=regularizers.l2(L2_REG),
            name="lstm_2"
        ),
        layers.Dropout(DROPOUT_RATE, name="dropout_2"),
        layers.Dense(VOCAB_SIZE, activation="softmax", name="output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model():
    # Initialize generators
    train_gen = data_generator(split="train", batch_size=BATCH_SIZE)
    val_gen = data_generator(split="val", batch_size=BATCH_SIZE)

    # Calculate steps
    def count_sequences(split):
        with open(f"single_file_{split}.txt") as f:
            return len(f.read().split(" / "))
    
    train_steps = count_sequences("train") // BATCH_SIZE
    val_steps = count_sequences("val") // BATCH_SIZE

    # Validate step counts
    if train_steps == 0 or val_steps == 0:
        raise ValueError(
            f"Insufficient data: train_steps={train_steps}, val_steps={val_steps}. "
            "Need more data or smaller batch size."
        )

   
    model = build_model()
    model.summary()

   
    checkpoint = callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    early_stop = callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor="val_loss"
    )

    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    # Save final model
    model.save("final_model.keras")
    return history

if __name__ == "__main__":
    
    if not all(os.path.exists(f"single_file_{split}.txt") for split in ["train", "val"]):
        raise FileNotFoundError("Missing preprocessed files. Run preprocessing.py first")
    
    train_model()