# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model import create_model
from DataSet import load_data  # your dataset pipeline

# -------------------------------
# CONFIG
# -------------------------------
MODEL_SAVE_PATH = 'saved_model/best_model.keras'
BATCH_SIZE = 16
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 10
LOW_LR = 1e-5

# -------------------------------
# TRAINING FUNCTION
# -------------------------------
def train_model():
    # 1️⃣ LOAD DATA
    train_gen, val_gen, test_gen, class_labels = load_data(batch_size=BATCH_SIZE)

    # 2️⃣ CREATE MODEL
    model, base_model = create_model()
    print("Initial model created.")
    model.summary()

    # 3️⃣ CALLBACKS
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

    # 4️⃣ INITIAL TRAINING
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_INITIAL,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    print("Initial training complete. Best model saved to:", MODEL_SAVE_PATH)

    # 5️⃣ FINE-TUNING
    # Unfreeze last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with low learning rate
    model.compile(optimizer=Adam(learning_rate=LOW_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    print("Fine-tuning complete. Best model saved to:", MODEL_SAVE_PATH)

    # Combine history dictionaries
    history.history.update(history_fine.history)
    return model, history, test_gen, class_labels


# -------------------------------
# SAFE ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    train_model()
