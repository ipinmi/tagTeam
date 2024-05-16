import numpy as np
import random


def viterbi_initialization(
    trans_matrix, emission_matrix, tag_index, vocab_index, input_sequence
):
    """
    Performs the initialization step of the Viterbi algorithm, by adding the probabilities of starting in each state

    Parameters:
        trans_matrix: numpy.ndarray
            a [Num_states, Num_states] matrix that stores the transition probabilities betwen tags
        emission_matrix: numpy.ndarray
            a [Num_states, vocab] matrix that stores the emission probabilities of each token given a tag
        input_sequence: list
            The input sequence of tokens
        vocab_index: dict
            A dictionary that maps each token to its corresponding index.
        tag_index: dict
            A dictionary that maps each tag to its corresponding index.

    Returns:
        path_prob_matrix: numpy.ndarray (initialized for the first observation)
            a [Num_states, Num_observations] matrix that stores the probability of the most likely path
        viterbi_path_matrix: numpy.ndarray (initialized for the first observation)
            a [Num_states, Num_observations] matrix that stores the best path for each observation
    """
    # the states are the unique tags in the training data
    states = list(tag_index.keys())
    start_idx = tag_index["<s>"]

    # the probabilities of starting in each state(Tag)
    initial_state_probs = np.zeros((1, len(states)))

    # a matrix with one column for each observation and one row for each state in the state graph
    path_prob_matrix = np.zeros((len(states), len(input_sequence)))
    viterbi_path_matrix = np.zeros((len(states), len(input_sequence)), dtype=int)

    for i in range(len(states)):
        initial_state_probs[0, i] = trans_matrix[start_idx, i]

        # if it is an out-of-vocabulary word, use the emission probability of the <unk> token
        if input_sequence[0] not in vocab_index.keys():
            path_prob_matrix[i, 0] = (
                initial_state_probs[0, i] * emission_matrix[i, vocab_index["<unk>"]]
            )
        # the probability of the first word in the input sequence
        # given the probaility of starting in each state
        else:
            path_prob_matrix[i, 0] = (
                initial_state_probs[0, i]
                * emission_matrix[i, vocab_index[input_sequence[0]]]
            )

    return path_prob_matrix, viterbi_path_matrix


def viterbi_forward(
    trans_matrix,
    emission_matrix,
    input_sequence,
    path_prob_matrix,
    viterbi_path_matrix,
    vocab_index,
):
    """
    Performs the recursion step of the Viterbi algorithm, which calculates the
    probability of the most likely path for the given input sequence.

    Parameters:
        trans_matrix: numpy.ndarray
            a [Num_states, Num_states] matrix that stores the transition probabilities betwen tags
        emission_matrix: numpy.ndarray
            a [Num_states, vocab] matrix that stores the emission probabilities of each token given a tag
        input_sequence: list
            The input sequence of tokens
        path_prob_matrix: numpy.ndarray (initialized for the first observation)
            a [Num_states, Num_observations] matrix that stores the probability of the most likely path
        viterbi_path_matrix: numpy.ndarray (initialized for the first observation)
            a [Num_states, Num_observations] matrix that stores the best path for each observation
        vocab_index: dict
            A dictionary that maps each token to its corresponding index.

    Returns:
        path_prob_matrix: numpy.ndarray
            a [Num_states, Num_observations] matrix that stores the probability of the most likely path
        viterbi_path_matrix: numpy.ndarray
            a [Num_states, Num_observations] matrix that stores the best path for each observation
    """
    num_states = path_prob_matrix.shape[0]
    for obv in range(1, len(input_sequence)):
        if input_sequence[obv] in vocab_index.keys():
            # iterate through all states and get the virterbi path probability
            # ((48,), (48, 48), (48, 1)) ==> (48,)
            path_prob_matrix[:, obv] = np.max(
                (
                    path_prob_matrix[:, obv - 1]
                    * trans_matrix.T
                    * emission_matrix[np.newaxis, :, vocab_index[input_sequence[obv]]].T
                ),
                1,
            )
        else:
            # if the word is not in the vocabulary, assign a random state
            path_prob_matrix[:, obv] = random.choice(np.arange(0, num_states))
        # get the best path for each state
        viterbi_path_matrix[:, obv] = np.argmax(
            (path_prob_matrix[:, obv - 1] * trans_matrix.T),
            1,
        )
    return path_prob_matrix, viterbi_path_matrix


def viterbi_backward(path_prob_matrix, viterbi_path_matrix, input_sequence, tag_index):
    """
    Performs the backtracking and termination step of the Viterbi algorithm,
    to get the sequence of tags that corresponds to the most likely path.

    Parameters:
        path_prob_matrix: numpy.ndarray
            A matrix with one column for each observation and one row for each state in the state graph.
        viterbi_path_matrix: numpy.ndarray
            A matrix that stores the best path for each state in the state graph.
        input_sequence: list
            The input sequence of words.
        tag_index: dict
            A dictionary that maps each tag to its corresponding index.

    Returns:
        best_path_tag: list
            the tag index for the most likely path.
        input_and_pred: list
            a list of lists that contaisn the input sequence and its predicted tags
    """

    num_obvs = len(input_sequence)

    # initialize the best path pointer
    best_path_pointer = np.empty(num_obvs, "B")
    # get the best path for the last observation
    best_path_pointer[-1] = np.argmax(path_prob_matrix[:, num_obvs - 1])

    # backtrack to get the best path for the remaining observations
    for i in reversed(range(1, num_obvs)):
        best_path_pointer[i - 1] = viterbi_path_matrix[best_path_pointer[i], i]

    # map the best path pointer to the corresponding tag
    idx2tag = {idx: tag for tag, idx in tag_index.items()}
    best_path_tag = [idx2tag[tag] for tag in best_path_pointer]

    # ensure that the input and the best path have the same length
    assert len(input_sequence) == len(best_path_tag)

    return best_path_tag
