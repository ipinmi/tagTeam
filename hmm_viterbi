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

def viterbi_forward(emission_matrix, transition_matrix, input_sentence, best_probs, best_paths):
    """
    
    Parameters:
    emission_matrix (numpy.ndarray): The emission probability matrix.
    transition_matrix (numpy.ndarray): The tag transition probability matrix.
    input_sentence (list): A list of words in the input sentence.
    best_probs (numpy.ndarray): The path probability matrix.
    best_paths (numpy.ndarray): The best path matrix.

    Returns:
    best_probs (numpy.ndarray): The updated path probability matrix.
    best_paths (numpy.ndarray): The updated best path matrix.

    """
    num_tags = best_probs.shape[0]
    for i in range(1, len(input_sentence)):
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None

            for k in range(num_tags):
                prob = best_probs[k,i-1] * transition_matrix[k,j] * emission_matrix[j,input_sentence[i]]
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i

    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, tag_counts):
    """
    
    Backward step of the Viterbi algorithm to retrieve the most probable tag sequence.

    Parameters:
    best_probs (numpy.ndarray): Matrix containing the highest probability of reaching each state at each time step.
    best_paths (numpy.ndarray): Matrix containing the most probable previous state for each state at each time step.
    tag_counts (dict): A dictionary containing the count of each tag.

    Returns:
    pred (list): Predicted sequence of tags corresponding to the input sentence.
    """
    m = best_paths.shape[1]
    z = [None] * m
    num_tags = best_probs.shape[0]

    best_prob_for_last_word = float('-inf')
    pred = [None] * m

    for k in range(num_tags):
        if best_probs[k, m - 1] > best_prob_for_last_word:
            best_prob_for_last_word = best_probs[k, m - 1]
            z[m - 1] = k
    pred[m - 1] = tag_counts[z[m - 1]]

    for i in range(m-1, -1, -1):
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i,i]
        pred[i - 1] = tag_counts[z[i - 1]]

    return pred
