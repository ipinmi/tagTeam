import numpy as np
from tag_transition import (
    tagCount_dictionaries,
    calculate_transition_probabilities,
    create_tag_transition_matrix,
)
from data import read_data_file, extract_tags
from emission import create_emission_matrix

import numpy as np

def state_initialization(emission_matrix, transition_matrix, tag_counts, input_sentence):
    """
    Initialize the state probability matrix viterbi[N,T] for the first observation.

    Parameters:
    emission_matrix (dict): The emission probability matrix.
    transition_matrix (dict): The tag transition probability matrix.
    tag_counts (dict): A dictionary containing the count of each tag.
    input_sentence (list): A list of words in the input sentence.

    Returns:
    best_probs (numpy.ndarray): The path probability matrix.
    best_paths (numpy.ndarray): The best path matrix.
    """

    # Create a path probability matrix viterbi[N,T]
    num_states = len(tag_counts)
    num_obvs = len(input_sentence)
    best_probs = np.zeros((num_states, num_obvs))
    best_paths = np.zeros((num_states, num_obvs), dtype=int)

    # Add the start state '<s>' to the tag counts if it's not already present
    if '<s>' not in tag_counts:
        tag_counts['<s>'] = len(tag_counts)

    # Find the index of the start state (if exists) in the tag counts
    start_index = tag_counts.get('<s>')

    # Putting the start tag as the first observation
    if start_index is not None:
        for tag in range(num_states):
            # Check if the (start_tag, tag) transition is possible
            if tag in transition_matrix.get(start_index, {}):
                best_probs[tag, 0] = transition_matrix[start_index].get(tag)  # Accessing transition probability using dictionary
            else:
                best_probs[tag, 0] = float("-inf")
    else:
        # Handle the case when the start state is not found in the tag counts
        raise ValueError("Start state '<s>' not found in tag counts")

    return best_probs, best_paths

def viterbi_forward(emission_matrix, transition_matrix, input_sentence, best_probs, best_paths):
    """
    Forward step of the Viterbi algorithm to compute the highest probability of reaching each state
    and the most probable path that leads to each state at each time step.

    Parameters:
    emission_matrix (dict): The emission probability matrix.
    transition_matrix (dict): The tag transition probability matrix.
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
                for word, prob in emission_matrix.get(input_sentence[i], []):
                    if j == word:
                        prob = best_probs[k, i - 1] * transition_matrix[k].get(j, 0) * prob
                        if prob > best_prob_i:
                            best_prob_i = prob
                            best_path_i = k

            if best_path_i is not None:
                best_probs[j, i] = best_prob_i
                best_paths[j, i] = best_path_i
            else:
                # If no valid path is found, set the probability to -inf
                best_probs[j, i] = float('-inf')

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
    num_words = best_paths.shape[1]
    pred = [None] * num_words

    # Find the index of the highest probability at the last time step
    max_prob_index = np.argmax(best_probs[:, num_words - 1])

    # Assign the tag corresponding to the highest probability at the last time step
    pred[num_words - 1] = list(tag_counts.keys())[max_prob_index]

    # Backtrack through the best_paths matrix to find the most probable sequence of tags
    for i in range(num_words - 1, 0, -1):
        if pred[i] is not None:
            pred[i - 1] = list(tag_counts.keys())[best_paths[max_prob_index, i]]
            max_prob_index = best_paths[max_prob_index, i]

    return pred

emission_matrix = create_emission_matrix("dummy.col")

#print(emission_matrix)

data = read_data_file("dummy.col")
tag_transition_counts, tag_counts = tagCount_dictionaries(data)
tag_transitions_prob, all_tags = calculate_transition_probabilities(
    tag_transition_counts, tag_counts
)
tag_transition_matrix = tag_transitions_prob

#print(tag_transition_matrix)

# Initialize Viterbi algorithm
input_sentence = ["<s>", "hi", "world"]
tag_counts = {tag: idx for idx, tag in enumerate(emission_matrix.keys())}
best_probs, best_paths = state_initialization(emission_matrix, tag_transition_matrix, tag_counts, input_sentence)
best_probs, best_paths = viterbi_forward(emission_matrix, tag_transition_matrix, input_sentence, best_probs, best_paths)
predicted_tags = viterbi_backward(best_probs, best_paths, tag_counts)

print("Predicted tags:", predicted_tags)
