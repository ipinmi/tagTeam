from data import extract_tokens_tags
from collections import defaultdict
import numpy as np


def tagCount_dictionaries(training_corpus):
    """
    Create dictionaries to store the count of transitions between tags and the count of each tag.

    Args:
        training_corpus (list): A list of lists where each inner list contains tuples of (word, tag).

    Returns:
        tag_transition_counts (dict): A dictionary that stores the count of transitions between two tags.
        tag_counts (dict): A dictionary that stores the count of each tag.
    """
    tag_transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    tags2idx = {}

    _, sentence_tags, _ = extract_tokens_tags(training_corpus)

    # Initialize "START" count
    tag_counts["<s>"] = 0

    for sentence_tags in sentence_tags:
        prev_tag = "<s>"  # Reset prev_tag for each sentence
        tag_counts[prev_tag] += 1  # Count occurrences of "START"
        for tag in sentence_tags:
            tag_transition_counts[(prev_tag, tag)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    for i, tok in enumerate(sorted(tag_counts.keys())):
        tags2idx[tok] = i

    return tag_transition_counts, tag_counts, tags2idx


def calculate_transition_probabilities(
    tag_transition_counts, tag_counts, smoothing_parameter=0.0001
):
    """
    Calculate transition probabilities based on tag transition counts and tag counts.

    Args:
        tag_transition_counts (dict): A dictionary that stores the count of transitions between two tags.
        tag_counts (dict): A dictionary that stores the count of each tag.

    Returns:
        tag_transition_probabilities (dict): A dictionary that stores transition probabilities for each tag.
        sorted_tags (list): A list of tags sorted in lexicographical order.
    """
    tag_transition_probabilities = defaultdict(dict)
    unique_tags = sorted(tag_counts.keys())
    num_tags = len(unique_tags)

    # Initialize transition probabilities for each tag
    for prev_tag in unique_tags:
        for tag in unique_tags:
            tag_transition_probabilities[prev_tag][tag] = smoothing_parameter / (
                tag_counts[prev_tag] + (smoothing_parameter * num_tags)
            )

    # Update transition probabilities based on observed transitions
    for (prev_tag, tag), count in tag_transition_counts.items():

        tag_transition_probabilities[prev_tag][tag] = max(
            tag_transition_probabilities[prev_tag][tag],
            (count + smoothing_parameter)
            / (tag_counts[prev_tag] + (smoothing_parameter * num_tags)),
        )

    # sort the tag probabilities lexographically
    for prev_tag in tag_transition_probabilities:
        tag_transition_probabilities[prev_tag] = dict(
            sorted(
                tag_transition_probabilities[prev_tag].items(),
                key=lambda item: item[0],
            )
        )

    sorted_tag_transition_prob = dict(sorted(tag_transition_probabilities.items()))

    return sorted_tag_transition_prob


def create_tag_transition_matrix(tag_transition_dict):
    """

    Parameters:
    tag_transition_dict: A dictionary that stores transition probabilities for each tag.

    Returns:
    tag_transition_matrix: A numpy array representing the tag transition matrix.
    """
    transition_list = []
    for key, value in tag_transition_dict.items():
        tag_list = []
        for tag, prob in value.items():
            tag_list.append(prob)
        transition_list.append(tag_list)

    tag_transition_matrix = np.array(transition_list)

    return tag_transition_matrix
