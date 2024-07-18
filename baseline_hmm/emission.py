# @Author: Hao-En Hsu
import numpy as np


def vocab2idx(data):
    """
    Create a dictionary that maps each token to its index in the vocabulary.

    Parameters:
    data (list): A list of list of sentences.

    Returns:
    vocab2idx (dict): A sorted dictionary that maps each token to its index in the vocabulary.
    """

    vocab = list(set(token for sentence in data for token, _ in sentence))

    # Add the unknown token to the vocabulary for out-of-vocabulary words
    vocab.append("<unk>")
    sorted_vocab = sorted(vocab)

    vocab2idx = {token: i for i, token in enumerate(sorted_vocab)}
    return vocab2idx


def calculate_emission_probabilities(data, vocab, smoothing_param=0.0001):
    """
    Create the emission matrix containing the probability of each observation given the tags.

    Parameters:
    filename (str): The filename of the input data.
    smoothing_param (float): A smoothing parameter (default value is 0.0001) to avoid zero probabilities.

    Returns:
    emission_matrix (dict): The sorted emission matrix containing the probability of each observation given the tags.
    """

    # Initialize dictionaries to store word-tag counts and tag counts
    word_tag_counts = {}
    tag_counts = {}

    tag_counts["<s>"] = 0
    word_tag_counts["<s>"] = {}

    # Count occurrences of each word-tag pair and each tag
    for sentence in data:
        for word, tag in sentence:
            if tag not in word_tag_counts:
                word_tag_counts[tag] = {}
            if word not in word_tag_counts[tag]:
                word_tag_counts[tag][word] = 0
            word_tag_counts[tag][word] += 1

            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1

    # Calculate probabilities for each word-tag pair with smoothing
    emission_probabilities = {}
    unique_tags = sorted(tag_counts.keys())

    # Initialize emission probabilities for each tag and token
    for tag in unique_tags:
        emission_probabilities[tag] = {}

    for tag in unique_tags:
        for token in vocab:
            emission_probabilities[tag][token] = smoothing_param / (
                tag_counts[tag] + (smoothing_param * len(vocab))
            )

    for tag, word_counts in word_tag_counts.items():
        if tag == "<s>":
            continue
        total_tag_count = tag_counts[tag]

        for token, count in word_counts.items():
            emission_probabilities[tag][token] = max(
                emission_probabilities[tag][token],
                (count + smoothing_param)
                / (total_tag_count + smoothing_param * len(vocab)),
            )

    # Sort the emission matrix by tag
    sorted_emission_probs = {
        tag: sorted(probabilities.items())
        for tag, probabilities in sorted(emission_probabilities.items())
    }

    return sorted_emission_probs


def create_emission_matrix(emission_probabilities):
    """
    Create the emission matrix from the dictionary containing the probability of each observation given the tags.

    Parameters:
    emission_probabilities (dict): A dictionary containing the emission probabilities for each tag.

    Returns:
    emission_matrix (np.array): A numpy array representing the emission matrix.
    """

    emission_list = []
    for tag, probs in emission_probabilities.items():
        tag_token_probs = []
        for word_prob_tuple in range(len(probs)):
            tag_token_probs.append(probs[word_prob_tuple][1])
        emission_list.append(tag_token_probs)

    emission_matrix = np.array(emission_list)

    return emission_matrix
