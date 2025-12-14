# model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES = 4  # adjust if needed
IMAGE_SIZE = (224, 224)

def create_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE):
    """
    Creates MobileNetV2 model with a custom dense head.
    """
    inputs = Input(shape=(*image_size, 3))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = False  # freeze initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model
