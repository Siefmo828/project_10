from DataSet import create_dataframe, create_generators, perform_stratified_split
from SuperMarketProject.project_10.code.train import BATCH_SIZE, IMAGE_SIZE
from utils import load_saved_model, evaluate_model_on_test

def evaluate_model():
    # --- Prepare test data ---
    full_df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(full_df)

    print("\n1. Preparing test generator...")
    _, _, test_gen = create_generators(
        df_train, df_val, df_test,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    print(f"Total test images: {test_gen.samples}")

    # --- Load trained model ---
    print("\n2. Loading the trained model...")
    model = load_saved_model("../saved_model/product_classifier")

    # --- Evaluate fully using utils ---
    class_labels = list(test_gen.class_indices.keys())
    evaluate_model_on_test(model, test_gen, class_labels, num_samples_predictions=6)

if __name__ == "__main__":
    evaluate_model()
