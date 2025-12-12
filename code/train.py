
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from dataset import create_dataframe, create_generators, perform_stratified_split
from model import create_cnn_model

IMAGE_SIZE = (256, 256)
INPUT_SHAPE = IMAGE_SIZE + (3,)
NUM_CLASSES = 4
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 10

SAVE_PATH = 'saved_model/best_model.h5'


def train_model():
    print("1. Loading data...")

    full_df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(full_df)

    train_gen, val_gen, test_gen = create_generators(
        df_train, df_val, df_test,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    print(f"Total training images: {train_gen.samples}")
    print(f"Total validation images: {val_gen.samples}")

    print("\n2. Creating model architecture...")
    model = create_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    print("\n3. Defining Callbacks...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    callbacks_list = [early_stop, checkpoint]

    print("\n4. Starting model training (Fit)...")
    history = model.fit(
        train_gen,
        epochs=MAX_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list
    )

    print("\nTraining complete. The best model has been saved to:", SAVE_PATH)

    return history


if __name__ == '__main__':
    train_model()
