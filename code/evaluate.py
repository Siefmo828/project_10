from DataSet import create_dataframe, create_generators, perform_stratified_split
from SuperMarketProject.project_10.code.train import BATCH_SIZE, IMAGE_SIZE
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model():

    full_df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(full_df)

    print("\n1. preparing tests...")
    _, _, test_gen = create_generators(
        df_train, df_val, df_test,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    print(f"Total test images: {test_gen.samples}")

    x_test = test_gen
    y_test = test_gen.classes

    print("\n2. importing the model...")
    model = load_model("../saved_model/product_classifier")


    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n3. Test set accuracy...")
    accuracy = accuracy_score(y_test, y_pred_classes)
    print("Test Accuracy:", accuracy)

    print("\n4. Classification report...")
    class_names = list(x_test.class_indices.keys())

    print(classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names
    ))

    print("\n5. Confusion matrix...")
    cm = confusion_matrix(y_test, y_pred_classes)

    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate_model()