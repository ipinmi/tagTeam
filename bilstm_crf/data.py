# @Author: Chibundum Adebayo

from collections import defaultdict
import numpy as np
import torch
import fasttext
import fasttext.util


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


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

        elif len(line.split()) == 3:
            word, lemma, tag = line.split()
            sentence.append((word, tag))
        elif len(line.split()) == 2:
            word, tag = line.split()
            sentence.append((word, tag))

    return data


def extract_tokens_tags(data):
    """

    Parameters:
    data (list): List of sentences in the following format:
    [(word1, tag1), (word2, tag2), ...] for each sentence.

    Returns:
    token_sequence (list): List of lists of words in the data.
    sentence_tags (list): List of lists of tags in the data.
    token_tag_list (list): List of tuples of (token_sequence, sentence_tags) for each sentence.
    all_tags_in_data (list): List of all tags in the data.

    """
    sentence_tags = [[tag for _, tag in sentence] for sentence in data]
    token_sequence = [[word for word, _ in sentence] for sentence in data]

    token_tag_list = []
    for i in range(len(token_sequence)):
        seq_tag_tuple = (token_sequence[i], sentence_tags[i])
        token_tag_list.append(seq_tag_tuple)

    # Flatten the list of lists
    all_tags_in_data = [tag for sentence in sentence_tags for tag in sentence]

    return token_sequence, sentence_tags, token_tag_list, all_tags_in_data


# Distribution of the tags
def tag_distribution(tag_sequences):
    """
    Returns the distribution of the tags in the data in the following format:
    {tag1: count1, tag2: count2, ...}
    """
    tag_distribution = defaultdict(int)
    for tag in tag_sequences:
        if tag in tag_distribution:
            tag_distribution[tag] += 1
        else:
            tag_distribution[tag] = 1
    return dict(sorted(tag_distribution.items()))


##### Using FastText Pretrained Embeddings #####


def get_pretrained_matrix(params, text_vocab_obj, model_path, save_path):
    """
    This function loads the pretrained FastText embeddings for the specified language code
    and creates a weights matrix for the vocabulary in the training data.

    Parameters:
    params (object): Parameters object containing the language code and word embedding dimension.
    text_vocab_obj (object): Vocabulary object for the training data.

    Returns:
    weight_matrix_tensor (torch.Tensor): Tensor containing the pretrained weights
    """
    # fasttext_model_path = f"{embedding_dir}/cc.{params.lang_code}.100.bin"
    # save_path = f"{embedding_dir}/{params.lang_code}_pretrained_weights.pt"

    # Load the downloaded FastText model
    pretrained_model = fasttext.load_model(model_path)

    # Get the vocabulary of the pretrained model
    pretrained_vocab = pretrained_model.get_words()

    # Get the embedding dimension from the training data
    embedding_dim = params.word_emb_dim

    # Initialize the weights matrix with random embeddings for all vocab words
    vocab_size = len(text_vocab_obj.idx2token)
    weights_matrix = np.random.randn(vocab_size, embedding_dim)

    # Iterate over the vocabulary and fetch corresponding embeddings
    for idx, word in enumerate(text_vocab_obj.idx2token):
        if idx < 2:  # Skip special tokens (pad, unk)
            continue
        if word in pretrained_vocab:
            embedding_vector = pretrained_model.get_word_vector(word)

            # Replace the random embeddings with pretrained embeddings
            weights_matrix[idx] = embedding_vector

        # Ensure that all embeddings are of the same dimension
        if len(weights_matrix[idx]) < embedding_dim:
            padding_amt = embedding_dim - len(weights_matrix[idx])
            weights_matrix[idx] = np.pad(
                weights_matrix[idx], (0, padding_amt), "constant"
            )

    # Convert the weights matrix to a tensor
    weight_matrix_tensor = torch.tensor(weights_matrix, dtype=torch.float32)

    # Save the tensor to the specified path
    torch.save(weight_matrix_tensor, save_path)
