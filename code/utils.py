# code/utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from tensorflow.keras.models import load_model


def plot_training_history(history, save_dir='results/'):
    """
    Plots training & validation loss and accuracy curves and saves them to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_path)
    plt.close()
    
    print(f"Training curves saved: {acc_path}, {loss_path}")


def plot_confusion_matrix(model, test_generator, class_labels, save_dir='results/'):
    """
    Evaluates the model on test data and plots a confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Predict classes
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(test_generator.classes, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Confusion matrix saved: {cm_path}")


def display_sample_predictions(model, test_generator, class_labels, num_samples=5, save_dir='results/'):
    """
    Displays and saves sample predictions from test data.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    test_generator.reset()
    x_batch, y_batch = next(test_generator)
    preds = model.predict(x_batch)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(x_batch))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_batch[i])
        plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[pred_classes[i]]}")
        plt.axis('off')
    
    save_path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Sample predictions saved: {save_path}")


def load_saved_model(model_path='saved_model/best_model.h5'):
    """
    Loads a Keras model from a given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first!")
    
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


# --- NEW FUNCTION ADDED ---
def evaluate_model_on_test(model, test_generator, class_labels, num_samples_predictions=5, save_dir='results/'):
    """
    Evaluates the model on test data: prints accuracy, classification report,
    plots confusion matrix, and shows sample predictions.
    """
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # --- Accuracy ---
    accuracy = accuracy_score(y_true, y_pred)
    print("\nTest Accuracy:", accuracy)

    # --- Classification report ---
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # --- Confusion matrix ---
    plot_confusion_matrix(model, test_generator, class_labels, save_dir=save_dir)

    # --- Sample predictions ---
    display_sample_predictions(model, test_generator, class_labels, num_samples=num_samples_predictions, save_dir=save_dir)
