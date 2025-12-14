import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

from tensorflow.keras.models import load_model


# -------------------------------------------------
# TRAINING CURVES
# -------------------------------------------------
def plot_training_history(history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    print("✔ Training curves saved")


# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
def plot_confusion_matrix(model, test_generator, class_labels, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    test_generator.reset()
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    plt.figure(figsize=(7, 6))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print("✔ Confusion matrix saved")


# -------------------------------------------------
# SAMPLE PREDICTIONS (FIXED IMAGE DISPLAY)
# -------------------------------------------------
def display_sample_predictions(
    model,
    test_generator,
    class_labels,
    num_samples=5,
    save_dir="results"
):
    os.makedirs(save_dir, exist_ok=True)

    test_generator.reset()
    x_batch, y_batch = next(test_generator)

    preds = model.predict(x_batch)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_batch, axis=1)

    plt.figure(figsize=(15, 4))

    for i in range(min(num_samples, len(x_batch))):
        plt.subplot(1, num_samples, i + 1)

        # 🔥 IMPORTANT FIX (undo MobileNetV2 preprocessing)
        img = (x_batch[i] + 1.0) / 2.0
        plt.imshow(img)

        plt.title(
            f"True: {class_labels[true_classes[i]]}\n"
            f"Pred: {class_labels[pred_classes[i]]}"
        )
        plt.axis("off")

    plt.savefig(os.path.join(save_dir, "sample_predictions.png"))
    plt.close()

    print("✔ Sample predictions saved")


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
def load_saved_model(model_path="saved_model/best_model.keras"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train the model first."
        )

    model = load_model(model_path)
    print(f"✔ Model loaded from {model_path}")
    return model


# -------------------------------------------------
# FULL TEST EVALUATION
# -------------------------------------------------
def evaluate_model_on_test(
    model,
    test_generator,
    class_labels,
    num_samples_predictions=5,
    save_dir="results"
):
    test_generator.reset()

    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✔ Test Accuracy: {acc:.4f}")

    # Classification report
    print("\n✔ Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_labels
    ))

    # Confusion matrix
    plot_confusion_matrix(
        model,
        test_generator,
        class_labels,
        save_dir
    )

    # Sample predictions
    display_sample_predictions(
        model,
        test_generator,
        class_labels,
        num_samples_predictions,
        save_dir
    )
