import pandas as pd # type: ignore
import os
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import numpy as np # type: ignore

# --- 1. CONFIGURATION AND MAPPING ---

# Set the path to the root folder containing the 25 product folders
# IMPORTANT: You must change this path to match your local setup!
DATA_DIR = '/home/siefmo/4th_1st/417/data/freiburg_groceries_dataset/images' 

# Define your custom 4-category mapping
CATEGORY_MAP = {
    # Pantry & Dry Goods (8 folders)
    'CEREAL': 'Pantry & Dry Goods', 'FLOUR': 'Pantry & Dry Goods', 
    'HONEY': 'Pantry & Dry Goods', 'NUTS': 'Pantry & Dry Goods', 
    'OIL': 'Pantry & Dry Goods', 'PASTA': 'Pantry & Dry Goods', 
    'RICE': 'Pantry & Dry Goods', 'SUGAR': 'Pantry & Dry Goods',
    
    # Beverage (6 folders)
    'COFFEE': 'Beverage', 'JUICE': 'Beverage', 'MILK': 'Beverage', 
    'SODA': 'Beverage', 'TEA': 'Beverage', 'WATER': 'Beverage',
    
    # Snack / Confectionery (4 folders)
    'CAKE': 'Snack / Confectionery', 'CANDY': 'Snack / Confectionery', 
    'CHIPS': 'Snack / Confectionery', 'CHOCOLATE': 'Snack / Confectionery', 
    
    # Canned / Preserved (6 folders)
    'BEANS': 'Canned / Preserved', 'CORN': 'Canned / Preserved', 
    'FISH': 'Canned / Preserved', 'JAM': 'Canned / Preserved', 
    'TOMATO_SAUCE': 'Canned / Preserved', 'VINEGAR': 'Canned / Preserved',
    
}

# The final 4 class names (used for output shape and generator class indices)
TARGET_CLASSES = sorted(list(set(CATEGORY_MAP.values())))
NUM_CLASSES = len(TARGET_CLASSES) # Should be 4


# --- 2. DATA COLLECTION AND SPLITTING ---

def create_dataframe(data_dir=DATA_DIR, category_map=CATEGORY_MAP):
    """Scans the directory and creates a DataFrame of file paths and labels."""
    filepaths = []
    coarse_labels = []

    for fine_class, coarse_class in category_map.items():
        class_path = os.path.join(data_dir, fine_class)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory not found for class: {fine_class} at {class_path}")
            continue

        for filename in os.listdir(class_path):
            # Only include image files (basic check)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(class_path, filename))
                coarse_labels.append(coarse_class)

    df = pd.DataFrame({'filepath': filepaths, 'coarse_label': coarse_labels})
    
    if df.empty:
        raise ValueError("No images found. Check DATA_DIR path and folder structure.")
    
    print(f"Total images found: {len(df)}")
    return df

def perform_stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Performs a stratified 70/15/15 split on the dataset."""
    
    # 1. Split into Training (70%) and Holdout (30%)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        df['filepath'], df['coarse_label'],
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=df['coarse_label'] # Essential for class balance
    )
    
    # Create DataFrames for Train and Holdout
    df_train = pd.DataFrame({'filepath': X_train, 'coarse_label': y_train})

    # 2. Split Holdout (30%) into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=test_ratio / (val_ratio + test_ratio), # 0.15 / 0.30 = 0.5
        random_state=random_seed,
        stratify=y_holdout # Essential for class balance
    )
    
    # Create DataFrames for Validation and Test
    df_val = pd.DataFrame({'filepath': X_val, 'coarse_label': y_val})
    df_test = pd.DataFrame({'filepath': X_test, 'coarse_label': y_test})
    
    print(f"Split results: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")
    return df_train, df_val, df_test


# --- 3. GENERATOR SETUP ---

def create_generators(df_train, df_val, df_test, batch_size=32, image_size=(256, 256)):
    """Creates the Keras ImageDataGenerators for each set."""
    
    # Data Augmentation for Training Set
    train_datagen = ImageDataGenerator(
        rescale=1./255,                 # Normalize pixel values
        rotation_range=30,              # Rotation
        width_shift_range=0.1,          # Translation (Shift)
        height_shift_range=0.1,         # Translation (Shift)
        shear_range=0.1, 
        zoom_range=0.1,                 # Zoom
        horizontal_flip=True,           # Flip
        brightness_range=[0.8, 1.2],    # Brightness/contrast adjustment
        fill_mode='nearest'
    )
    
    # Only Rescaling for Validation and Test Sets (NO AUGMENTATION)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # --- Training Generator ---
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', # Required for Categorical Cross-Entropy loss
        classes=TARGET_CLASSES,   # Ensures consistent class indexing
        shuffle=True
    )
    
    # --- Validation Generator ---
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', 
        classes=TARGET_CLASSES,
        shuffle=False # No need to shuffle validation data
    )
    
    # --- Test Generator ---
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=False # Must NOT shuffle test data for correct evaluation
    )
    
    return train_generator, val_generator, test_generator

# --- 4. MAIN EXPOSED FUNCTION ---

def load_data(batch_size=32, image_size=(256, 256)):
    """
    Main function to load and prepare data for training.

    Returns: train_generator, val_generator, test_generator, class_labels
    """
    df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(df)
    
    train_gen, val_gen, test_gen = create_generators(
        df_train, df_val, df_test, batch_size, image_size
    )
    
    # The class labels are stored in the generator, but returning the list is helpful
    return train_gen, val_gen, test_gen, TARGET_CLASSES

# Example usage (for testing this script directly):
if __name__ == '__main__':
    try:
        train_gen, val_gen, test_gen, class_labels = load_data(batch_size=32)
        print("\nData loading successful!")
        print(f"Final Class Labels (in order): {class_labels}")
        print(f"Number of classes: {NUM_CLASSES}")
        
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("\nACTION REQUIRED: Please update the 'DATA_DIR' variable at the top of dataset.py.")