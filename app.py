# app.py
import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ CONFIGURATION
# -------------------------------
MODEL_PATH = "saved_model/best_model.keras"
RESULTS_DIR = "results/"
IMAGE_SIZE = (224, 224)

CATEGORY_MAP = {
    'CEREAL': 'Pantry & Dry Goods', 'FLOUR': 'Pantry & Dry Goods', 
    'HONEY': 'Pantry & Dry Goods', 'NUTS': 'Pantry & Dry Goods', 
    'OIL': 'Pantry & Dry Goods', 'PASTA': 'Pantry & Dry Goods', 
    'RICE': 'Pantry & Dry Goods', 'SUGAR': 'Pantry & Dry Goods',
    'COFFEE': 'Beverage', 'JUICE': 'Beverage', 'MILK': 'Beverage', 
    'SODA': 'Beverage', 'TEA': 'Beverage', 'WATER': 'Beverage',
    'CAKE': 'Snack / Confectionery', 'CANDY': 'Snack / Confectionery', 
    'CHIPS': 'Snack / Confectionery', 'CHOCOLATE': 'Snack / Confectionery', 
    'BEANS': 'Canned / Preserved', 'CORN': 'Canned / Preserved', 
    'FISH': 'Canned / Preserved', 'JAM': 'Canned / Preserved', 
    'TOMATO_SAUCE': 'Canned / Preserved', 'VINEGAR': 'Canned / Preserved',
}

CLASS_NAMES = sorted(list(set(CATEGORY_MAP.values())))

# -------------------------------
# 2️⃣ LOAD MODEL
# -------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# -------------------------------
# 3️⃣ STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Product Classifier", layout="wide")
st.title("🛒 Product Category Classifier")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Evaluation", "Category Map"])

# -------- Predict Tab --------
with tab1:
    st.write("Upload an image to predict its product category.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=400)

        # Preprocess & predict
        img_resized = img.resize(IMAGE_SIZE)
        x = np.array(img_resized)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        predicted_idx = np.argmax(preds)
        predicted_class = CLASS_NAMES[predicted_idx]
        probability = preds[0][predicted_idx]

        st.success(f"Predicted category: **{predicted_class}**")
        st.info(f"Probability: {probability:.2f}")

        # Nice horizontal bar chart
        prob_df = pd.DataFrame({
            "Category": CLASS_NAMES,
            "Probability": preds[0]
        }).sort_values("Probability", ascending=True)  # sort ascending for horizontal bars

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(prob_df["Category"], prob_df["Probability"], color="skyblue")
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.2f}", color='blue', va='center')
        st.pyplot(fig)


# -------- Evaluation Tab --------
with tab2:
    st.header("📊 Model Evaluation Results")
    eval_images = ["Accuracy.png", "loss.png", "confusion_matrix.png", "sample.png"]
    for img_name in eval_images:
        img_path = os.path.join(RESULTS_DIR, img_name)
        if os.path.exists(img_path):
            st.image(img_path, caption=img_name, width=600)
        else:
            st.warning(f"{img_name} not found in {RESULTS_DIR}/")

# -------- Category Map Tab --------
with tab3:
    st.header("🗂 Category Map")
    st.write("Shows how items are mapped to main categories.")
    df_map = pd.DataFrame(list(CATEGORY_MAP.items()), columns=["Item", "Category"])
    st.dataframe(df_map)
