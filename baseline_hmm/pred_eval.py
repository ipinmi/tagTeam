# @Author: Chibundum Adebayo, Hao-En Hsu
from hmm import *


def predict_tags_for_sentence(
    tag_transition_matrix, emission_matrix, tag_index, vocab_index, input_sentence
):
    """
    Predict tags for a given input sentence using the Viterbi algorithm.

    Parameters:
        emission_matrix (numpy.ndarray): The emission probability matrix.
        tag_transition_matrix (numpy.ndarray): The tag transition probability matrix.
        tag_index (dict): A dictionary mapping each tag to its index.
        vocab_index (dict): A dictionary mapping each word to its index.
        input_sentence (list): A list of words in the input sentence.

    Returns:
        predicted_tags (list): Predicted sequence of tags corresponding to the input sentence.
    """

    # Initialize Viterbi Path Probability matrix and Path matrix
    path_prob_matrix, viterbi_path_matrix = viterbi_initialization(
        tag_transition_matrix, emission_matrix, tag_index, vocab_index, input_sentence
    )
    best_probs, best_paths = viterbi_forward(
        tag_transition_matrix,
        emission_matrix,
        input_sentence,
        path_prob_matrix,
        viterbi_path_matrix,
        vocab_index,
    )
    predicted_tags = viterbi_backward(best_probs, best_paths, input_sentence, tag_index)

    return predicted_tags


def save_tagged_sentences(
    filename, sentences, emission_matrix, tag_transition_matrix, tag_index, vocab_index
):
    """
    Collect the predicted tags over the entire dataset and save the predictions to a file.

    Returns:
        all_predicted_tags (list): A merged list of all predicted tags for the dataset.
    """

    # Create lists to store predicted tags for each sentence
    all_predicted_tags = []

    # Create prediction file for the current dataset
    if "test" in filename:
        prediction_file = "test_predictions.txt"
    else:
        prediction_file = "dev_predictions.txt"

    with open(prediction_file, "a") as file:
        # Iterate over each sentence in the data
        for sentence in sentences:
            # Predict tags for the current sentence using the Viterbi algorithm
            predicted_sentence_tags = predict_tags_for_sentence(
                tag_transition_matrix, emission_matrix, tag_index, vocab_index, sentence
            )
            token_tag_tuple = list(zip(sentence, predicted_sentence_tags))

            # Write the token-tag tuples to the file
            for token, tag in token_tag_tuple:
                file.write(f"{token}\t{tag}\n")
            file.write("\n")  # Add a new line after each sentence

            # Store the predicted tags for this sentence
            all_predicted_tags.extend(predicted_sentence_tags)

    return all_predicted_tags
