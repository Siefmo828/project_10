# README: Supermarket Product Classification 🛒

This repository contains the code and resources for a Convolutional Neural Network (CNN) project aimed at classifying supermarket products into broad categories.

***

## 1. Project Description

This project implements a CNN to classify grocery products from images into four primary categories: **Beverage**, **Canned / Preserved**, **Pantry & Dry Goods**, and **Snack / Confectionery**.

The primary goal is to develop an efficient model capable of recognizing product categories despite variations in lighting, packaging design, and scene clutter. 
This supports applications like automated inventory management and smart checkout systems, as outlined in the project brief.

**Key features include:**
* A customized data pipeline (`dataset.py`) for **stratified splitting** of the dataset into **Train (70%), Validation (15%), and Test (15%)** sets.
* **Data Augmentation** (rotation, zoom, flip) applied to the training set to increase model robustness and prevent overfitting.
* Training a CNN using the **Adam** optimizer and **Categorical Cross-Entropy** loss, with **Early Stopping** and **Model Checkpointing** to optimize training.

***

## 2. Dataset

The project uses the **Freiburg Groceries Dataset**.

| Detail | Value |
| :--- | :--- |
| **Source** | University of Freiburg |
| **Total Images** | $\approx 5,000$ |
| **Project Classes** | 4 (consolidated from 25) |

### Dataset Link

The dataset is publicly available for download:

* **URL:** [Freiburg Groceries Dataset]([http://ais.informatik.uni-freiburg.de/projects/datasets/bosch.html#groceries](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/))
* (Look for the link to the image dataset zip file on this page.)

## 3. How to Install Dependencies

This project requires Python and several standard scientific and deep learning libraries.

### Step 1: Create a Virtual Environment (Recommended)

Use a Python virtual environment to manage dependencies:

```bash
# Create the environment
python3 -m venv venv 
```
# Activate the environment
# On macOS/Linux:
```bash
source venv/bin/activate
```
# On Windows (Command Prompt):
```
# venv\Scripts\activate
```

Step 2: Install Required Libraries
Install the necessary packages using pip:
```
pip install tensorflow pandas scikit-learn numpy
```
4. How to Run the Code
Once dependencies are installed and the DATA_DIR path is correct in dataset.py, follow these steps:

1. Training the Model
The train.py script loads the data, builds the CNN (from model.py), and starts the training process, saving the best-performing model to saved_model/best_model.h5.
```bash
python code/train.py
```
2. Evaluating the Model
The evaluate.py script loads the saved model and runs it against the unbiased Test set to generate final performance metrics.
```bash
python code/evaluate.py
```
