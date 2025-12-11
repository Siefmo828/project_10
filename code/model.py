# code/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers

def create_cnn_model(input_shape=(256, 256, 3), num_classes=4):
    """
    Defines and compiles the CNN model for 4-class product classification.
    
    Args:
        input_shape (tuple): The shape of the input images (Height, Width, Channels).
        num_classes (int): The number of output classes (4: Beverage, Canned, Pantry, Snack).
        
    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()

    # --- Feature Extraction Layers ---
    
    # Block 1: Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Block 2: Deeper Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 3: Deepest Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Introduce Dropout (a form of regularization) to help prevent overfitting
    model.add(Dropout(0.25))

    # --- Classification Layers (Head) ---
    
    # Flatten the 3D feature maps into a 1D vector for the Dense layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(512, activation='relu'))
    
    # Fully Connected Layer 2 (Output Layer)
    # Uses 'softmax' activation for multi-class probability output
    model.add(Dense(num_classes, activation='softmax'))

    # --- Compilation ---
    
    # Compile the model with required parameters
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

if __name__ == '__main__':
    # Example usage for testing the model architecture
    model = create_cnn_model()