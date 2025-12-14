from utils import load_saved_model, evaluate_model_on_test
from DataSet import create_dataframe, create_generators, perform_stratified_split
from train import BATCH_SIZE, IMAGE_SIZE

def evaluate_model():
    # --- Prepare test data ---
    df = create_dataframe()
    df_train, df_val, df_test = perform_stratified_split(df)
    _, _, test_gen = create_generators(df_train, df_val, df_test, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # --- Load trained 4-class model ---
    model = load_saved_model("saved_model/best_model.keras")

    # --- Evaluate ---
    class_labels = sorted(list(set([row for row in df_test['coarse_label']])))
    evaluate_model_on_test(model, test_gen, class_labels, num_samples_predictions=6)

if __name__ == "__main__":
    evaluate_model()
