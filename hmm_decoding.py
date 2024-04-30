import numpy as np


def state_initialization(
    emission_matrix, transition_matrix, tag_counts, input_sentence
):
    """
    Initialize the state probability matrix viterbi[N,T] for the first observation.

    Parameters:
    emission_matrix (numpy.ndarray): The emission probability matrix.
    transition_matrix (numpy.ndarray): The tag transition probability matrix.
    tag_counts (dict): A dictionary containing the count of each tag.
    input_sentence (list): A list of words in the input sentence.

    Returns:
    best_probs (numpy.ndarray): The path probability matrix.
    best_paths (numpy.ndarray): The best path matrix.
    """

    # create a path probability matrix viterbi[N,T]

    num_states = len(tag_counts)
    num_obvs = len(input_sentence)
    best_probs = np.zeros((num_states, num_obvs))
    best_paths = np.zeros((num_states, num_obvs), dtype=int)

    start_index = list(transition_matrix.keys()).index("<s>")

    # putting the start tag as the first observation
    for tag in range(num_states):
        # check if the (start_tag, tag) transition is possible
        if transition_matrix[start_index, tag] == 0:
            best_probs[tag, 0] = float("-inf")
        else:
            # calculate the probability of the path
            best_probs[tag, 0] = (
                transition_matrix[start_index, tag]
                * emission_matrix[tag, input_sentence[0]]
            )

    return best_probs, best_paths
