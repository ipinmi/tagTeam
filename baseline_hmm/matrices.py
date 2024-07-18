# @Author: Chibundum Adebayo
from data import read_data_file
from baseline_hmm.transition import *
from emission import *
import numpy as np


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


@run_once
def build_matrices():
    """
    This function builds the transition and emission matrices from the training data
    and saves them to a numpy pickle file.
    It is only run once to avoid recomputing the matrices every time the program is run.
    """
    # Read the training data
    filepath = "/mount/studenten/team-lab-cl/pos/train.col"
    train_data = read_data_file(filepath)

    smothing_param = 0.0001  # Smoothing parameter to avoid zero probabilities

    # Create the vocabulary and map each token to its index
    vocabulary_index = vocab2idx(train_data)

    # Calculate the emission probabilities and create the emission matrix
    emission_prob_dict = calculate_emission_probabilities(
        train_data, vocabulary_index, smothing_param
    )
    emission_matrix = create_emission_matrix(emission_prob_dict)

    # Create the tag transition matrix and map each tag to its index
    tag_transition_counts, tag_counts, tags2idx = tagCount_dictionaries(train_data)

    transition_prob_dict = calculate_transition_probabilities(
        tag_transition_counts, tag_counts, smothing_param
    )

    transition_matrix = create_tag_transition_matrix(transition_prob_dict)

    # Save the emission and transition matrices
    np.save("transition.npy", transition_matrix)
    np.save("emission.npy", emission_matrix)

    # Save the vocabulary and tags indices
    np.save("vocab_index.npy", vocabulary_index)
    np.save("tag_index.npy", tags2idx)
