import numpy as np
from data import read_data_file, extract_tokens_tags
from matrices import build_matrices
from pred_eval import *
from evaluation import Evaluation


def run_model():
    indices_and_matrices = build_matrices()

    # Load all the required matrices and dictionaries
    tag_transition_matrix = np.load("transition.npy", allow_pickle=True)
    emission_matrix = np.load("emission.npy", allow_pickle=True)
    vocab_index = np.load("vocab_index.npy", allow_pickle=True).item()
    tag_index = np.load("tag_index.npy", allow_pickle=True).item()

    # Read the appropriate experiment file to get the gold standard tags
    filename = "dataset/test.col"
    gold_standard_data = read_data_file(filename)
    sentences, _, gold_standard_tags = extract_tokens_tags(gold_standard_data)

    all_predicted_tags = save_tagged_sentences(
        filename,
        sentences,
        emission_matrix,
        tag_transition_matrix,
        tag_index,
        vocab_index,
    )

    # Evaluate the predicted tags against the gold standard tags
    evaluator = Evaluation(all_predicted_tags, gold_standard_tags)

    # Calculate precision, recall, and F1 score for all predicted tags
    precision, recall, f1_score = evaluator.precision_recall_fScore()

    # Print the results for all predicted tags

    with open("evaluation_results.txt", "a") as file:
        if "test" in filename:
            file.write("Evaluation results for the test data:\n")
        else:
            file.write("Evaluation results for the dev data:\n")
        file.write(f"Precision {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1_score}\n")
        file.write("\n")


if __name__ == "__main__":
    run_model()
