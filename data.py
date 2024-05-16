def read_data_file(file_name):
    """
    Reading the CONNLU format file in the following format:
    [(word1, tag1), (word2, tag2), ...] for each sentence.
    """
    with open(file_name, "r") as file:
        lines = file.readlines()

    data = []
    sentence = []
    for line in lines:
        if line == "\n":
            data.append(sentence)
            sentence = []
        else:
            word, tag = line.split()
            sentence.append((word, tag))

    return data


def extract_tokens_tags(data):
    """
    Extracting the tags from the data for evaluation in the following format:
    [tag1, tag2, ...] for each sentence.
    """
    sentence_tags = [[tag for _, tag in sentence] for sentence in data]
    token_sequence = [[word for word, _ in sentence] for sentence in data]

    # Flatten the list of lists
    all_tags_in_data = [tag for sentence in sentence_tags for tag in sentence]

    return token_sequence, sentence_tags, all_tags_in_data
