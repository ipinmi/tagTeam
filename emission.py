from data import read_data_file

def create_emission_matrix(filename, smoothing_param=0.0001):
    """
    Create the emission matrix containing the probability of each observation given the tags.

    Parameters:
    filename (str): The filename of the input data.
    smoothing_param (float): A smoothing parameter (default value is 0.0001) to avoid zero probabilities.

    Returns:
    emission_matrix (dict): The emission matrix containing the probability of each observation given the tags.
    """
    # Read data from file
    data = read_data_file(filename)

    # Initialize dictionaries to store word-tag counts and tag counts
    word_tag_counts = {}
    tag_counts = {}

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
    emission_matrix = {}
    for tag, word_counts in word_tag_counts.items():
        total_tag_count = tag_counts[tag]
        emission_matrix[tag] = {}
        for word, count in word_counts.items():
            emission_matrix[tag][word] = (count + smoothing_param) / (total_tag_count + smoothing_param * len(word_counts))

    # Sort the emission matrix by tag
    sorted_emission_matrix = {tag: sorted(probabilities.items()) for tag, probabilities in sorted(emission_matrix.items())}

    return sorted_emission_matrix


emission_matrix = create_emission_matrix("dummy.col")
print(emission_matrix)
