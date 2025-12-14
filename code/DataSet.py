import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1️⃣ CONFIGURATION
# -------------------------------
DATA_DIR = '/home/siefmo/4th_1st/417/data/freiburg_groceries_dataset/images/'

CATEGORY_MAP = {
    # Pantry & Dry Goods
    'CEREAL': 'Pantry & Dry Goods', 'FLOUR': 'Pantry & Dry Goods', 
    'HONEY': 'Pantry & Dry Goods', 'NUTS': 'Pantry & Dry Goods', 
    'OIL': 'Pantry & Dry Goods', 'PASTA': 'Pantry & Dry Goods', 
    'RICE': 'Pantry & Dry Goods', 'SUGAR': 'Pantry & Dry Goods',
    
    # Beverage
    'COFFEE': 'Beverage', 'JUICE': 'Beverage', 'MILK': 'Beverage', 
    'SODA': 'Beverage', 'TEA': 'Beverage', 'WATER': 'Beverage',
    
    # Snack / Confectionery
    'CAKE': 'Snack / Confectionery', 'CANDY': 'Snack / Confectionery', 
    'CHIPS': 'Snack / Confectionery', 'CHOCOLATE': 'Snack / Confectionery', 
    
    # Canned / Preserved
    'BEANS': 'Canned / Preserved', 'CORN': 'Canned / Preserved', 
    'FISH': 'Canned / Preserved', 'JAM': 'Canned / Preserved', 
    'TOMATO_SAUCE': 'Canned / Preserved', 'VINEGAR': 'Canned / Preserved',
}

TARGET_CLASSES = sorted(list(set(CATEGORY_MAP.values())))
NUM_CLASSES = len(TARGET_CLASSES)
IMAGE_SIZE = (224, 224)

# -------------------------------
# 2️⃣ CREATE DATAFRAME
# -------------------------------
def create_dataframe(data_dir=DATA_DIR, category_map=CATEGORY_MAP):
    filepaths = []
    coarse_labels = []

    for fine_class, coarse_class in category_map.items():
        class_path = os.path.join(data_dir, fine_class)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory not found for class: {fine_class}")
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(class_path, filename))
                coarse_labels.append(coarse_class)
    
    df = pd.DataFrame({'filepath': filepaths, 'coarse_label': coarse_labels})
    if df.empty:
        raise ValueError("No images found. Check DATA_DIR path and folder structure.")
    
    print(f"Total images found: {len(df)}")
    return df

# -------------------------------
# 3️⃣ STRATIFIED SPLIT
# -------------------------------
def perform_stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        df['filepath'], df['coarse_label'],
        test_size=(val_ratio + test_ratio),
        stratify=df['coarse_label'],
        random_state=random_seed
    )
    
    df_train = pd.DataFrame({'filepath': X_train, 'coarse_label': y_train})
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y_holdout,
        random_state=random_seed
    )
    
    df_val = pd.DataFrame({'filepath': X_val, 'coarse_label': y_val})
    df_test = pd.DataFrame({'filepath': X_test, 'coarse_label': y_test})
    
    print(f"Split results: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")
    return df_train, df_val, df_test

# -------------------------------
# 4️⃣ GENERATORS WITH AUGMENTATION
# -------------------------------
def create_generators(df_train, df_val, df_test, batch_size=32, image_size=IMAGE_SIZE):
    # --- Training Augmentation ---
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    # --- Validation / Test ---
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # --- Generators ---
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col='filepath',
        y_col='coarse_label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

# -------------------------------
# 5️⃣ MAIN LOAD FUNCTION
# -------------------------------
def load_data(batch_size=32, image_size=IMAGE_SIZE):
    df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(df)
    train_gen, val_gen, test_gen = create_generators(df_train, df_val, df_test, batch_size, image_size)
    return train_gen, val_gen, test_gen, TARGET_CLASSES

# -------------------------------
# 6️⃣ TEST RUN
# -------------------------------
if __name__ == '__main__':
    try:
        train_gen, val_gen, test_gen, class_labels = load_data(batch_size=32)
        print("\nData loading successful!")
        print(f"Class labels: {class_labels}")
        print(f"Number of classes: {NUM_CLASSES}")
    except ValueError as e:
        print(f"Error loading data: {e}")
