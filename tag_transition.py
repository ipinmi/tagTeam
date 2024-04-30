from data import extract_tags
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

    sentence_tags, _ = extract_tags(training_corpus)

    # Initialize "START" count
    tag_counts["<s>"] = 0

    for sentence_tags in sentence_tags:
        prev_tag = "<s>"  # Reset prev_tag for each sentence
        tag_counts[prev_tag] += 1  # Count occurrences of "START"
        for tag in sentence_tags:
            tag_transition_counts[(prev_tag, tag)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    return tag_transition_counts, tag_counts


def calculate_transition_probabilities(tag_transition_counts, tag_counts):
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

    for (prev_tag, tag), count in tag_transition_counts.items():
        if prev_tag not in tag_transition_probabilities:
            tag_transition_probabilities[prev_tag] = {}

        if tag not in tag_transition_probabilities[prev_tag]:
            tag_transition_probabilities[prev_tag][tag] = count / tag_counts[prev_tag]
        else:
            # Update probability if the transition already exists
            tag_transition_probabilities[prev_tag][tag] = max(
                tag_transition_probabilities[prev_tag][tag],
                count / tag_counts[prev_tag],
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
    sorted_tags = list(sorted_tag_transition_prob.keys())

    return sorted_tag_transition_prob, sorted_tags


def create_tag_transition_matrix(transition_counts, tag_counts, smoothing_parameter=0):
    """

    Parameters:
    transition_counts: A dictionary that stores the count of transitions between two tags.
    tag_counts: A dictionary that stores the count of each tag.
    smoothing_parameter: A smoothing parameter (default value is 0) to avoid zero probabilities.

    Returns:
    tag_transition_matrix: A numpy array representing the tag transition matrix.
    """
    unique_tags = sorted(tag_counts.keys())
    num_tags = len(unique_tags)

    # initialize the transition matrix
    tag_transition_matrix = np.zeros((num_tags, num_tags))

    for i in range(num_tags):
        for j in range(num_tags):
            count = 0

            key = (
                unique_tags[i],
                unique_tags[j],
            )  # key is a tuple of the possible previous and next tags
            if key in transition_counts:
                count = transition_counts[key]
            count_prev_tag = tag_counts[unique_tags[i]]

            tag_transition_matrix[i, j] = (count + smoothing_parameter) / (
                count_prev_tag + smoothing_parameter * num_tags
            )

    return tag_transition_matrix
