# @Author: Chibundum Adebayo

import torch
import random
from collections import defaultdict
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary(object):
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        """
        This class creates the vocabulary for a dataset. It maps tokens to indices and vice versa.
        As well as, appending to the vocabulary dictionary.

        Parameters:
        pad_token (str): The padding token.
        unk_token (str): The unknown token.

        Methods:
        token_to_idx: Maps a tag/token to an index.
        idx_to_token: Maps an index to a tag/token.
        unk_idx: Returns the index of the unknown token.
        vocab_size: Returns the size of the vocabulary.
        pad_idx: Returns the index of the padding token.
        add_token_sequence: Adds a sequence of tokens to the vocabulary.
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        # Initialize the token to index and index to token dictionaries with the pad and unk tokens.
        self.idx2token = [pad_token, unk_token]
        self.token2idx = defaultdict(lambda: self.idx2token.index(self.pad_token))
        # Set the indices for the pad and unk tokens.
        self.token2idx[pad_token] = 0
        self.token2idx[unk_token] = 1

    def token_to_idx(self, token: str) -> int:
        if token not in self.token2idx:
            return self.token2idx[self.unk_token]
        else:
            return self.token2idx[token]

    def idx_to_token(self, idx: int) -> str:
        return self.idx2token[idx]

    @property
    def unk_idx(self):
        return self.token_to_idx(self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    @property
    def pad_idx(self):
        return self.token_to_idx(self.pad_token)

    def add_token_sequence(self, token_sequence: List[str]):
        for token in token_sequence:
            if token not in self.token2idx:
                self.token2idx[token] = self.vocab_size
                self.idx2token.append(token)


class POS_Dataset(object):
    """
    This class creates a dataset for a sequence tagging task. It reads the data and creates the vocabulary.

    Methods:
    read_dataset: Reads the input data and converts it to a tensor.
    add_test_dev_set: Adds the test and dev sets to the dataset.
    get_vocabulary: Returns the vocabulary for the specified vocabulary.
    sentence_to_array: Converts a sentence to an array of integers.
    array_to_sentence: Converts an array of integers to a sentence.
    shuffle_train_data: Shuffles the training data.
    """

    def __init__(
        self,
        train_data: List[Tuple[List, List]],
        test_data: List[Tuple[List, List]],
        dev_data: List[Tuple[List, List]],
    ):
        # Initialize the vocabularies using the Vocabulary class.
        self.word_vocab = Vocabulary()
        self.char_vocab = Vocabulary()
        self.tags_vocab = Vocabulary()

        train_examples = self.read_dataset(train_data, add_to_vocabularies=True)

        self.data = {
            "train": {
                "examples": train_examples,
            },
            "dev": {},
            "test": {},
        }

        # Add the test and dev sets to the data dictionary.
        self.add_test_dev_set(test_data, "test")
        self.add_test_dev_set(dev_data, "dev")

    def read_dataset(
        self, input_data: List[Tuple[List, List]], add_to_vocabularies: bool
    ):
        """
        Convert the input data to a tensor and save the length of each example.
        Also, creates the character level input tensor.

        Parameters:
        input_data (list): List of tuples of (input_list, target_list) for each sentence.
        add_to_vocabularies (bool): Whether to add the tokens to the vocabularies based on the split.

        Returns:
        examples (list): List of dictionaries containing the word array, word length, char input list, and tag sequence array.
        """
        # Convert each example to a tensor and save the length
        examples = []
        for input_list, target_list in input_data:
            assert len(input_list) == len(
                target_list
            ), "Input and target lengths have to match"

            # Set to false if adding test or dev set tokens to vocabulary.
            if add_to_vocabularies:
                self.word_vocab.add_token_sequence(input_list)
                self.tags_vocab.add_token_sequence(target_list)

            # Convert each word in the sentence into an integer.
            input_array = self.sentence_to_array(input_list, vocabulary="word")

            # Convert each word in the sentence into a list of integers per character.
            char_inputs = []
            for word in input_list:
                char_list = list(word)
                if add_to_vocabularies:
                    self.char_vocab.add_token_sequence(char_list)

                char_array = self.sentence_to_array(char_list, vocabulary="char")
                char_inputs.append(char_array)
            target_array = self.sentence_to_array(target_list, vocabulary="tags")
            examples.append(
                {
                    "word_array": input_array,
                    "word_length": len(input_array),
                    "char_input_list": char_inputs,
                    "tag_seq_array": target_array,
                }
            )
        return examples

    def add_test_dev_set(
        self,
        data: List[Tuple[List, List]],
        split: str,
    ):
        """
        Create the test and dev sets and add them to the data dictionary.

        Parameters:
        data (list): List of tuples of (input_list, target_list) for each sentence.
        split (str): The split to add the data to.

        """
        if split == "test":
            test_examples = self.read_dataset(data, add_to_vocabularies=False)
            self.data["test"]["examples"] = test_examples

        elif split == "dev":
            dev_examples = self.read_dataset(data, add_to_vocabularies=False)
            self.data["dev"]["examples"] = dev_examples

    def get_vocabulary(self, vocab_name: str) -> Vocabulary:
        """
        Get the vocabulary based on the specified vocabulary.
        """
        if vocab_name == "word":
            vocab = self.word_vocab
        elif vocab_name == "char":
            vocab = self.char_vocab
        elif vocab_name == "tags":
            vocab = self.tags_vocab
        else:
            raise ValueError(
                "The given vocabulary name does not exist: {}".format(vocab_name)
            )
        return vocab

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Map each word in a sentence to an integer based on the vocabulary.

        Parameters:
        sentence (list): List of words.
        vocabulary (str): The vocabulary to use.

        Returns:
        sentence_array (list): List of integers representing the words from the vocabulary.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = []
        for word in sentence:
            sentence_array.append(vocab.token_to_idx(word))
        return sentence_array

    def array_to_sentence(
        self, sentence_array: List[int], vocabulary: str
    ) -> List[str]:
        """
        Maps each integer in a sentence array to a word based on the vocabulary.

        Parameters:
        sentence_array (list): List of integers representing the words.
        vocabulary (str): The vocabulary to use.

        Returns:
        sentence (list): List of words from the vocabulary.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [
            vocab.idx_to_token(token_idx) for token_idx in sentence_array.squeeze(dim=0)
        ]

    def shuffle_train_data(self):
        """
        Shuffles the training data to prevent the model from learning the order of the data.
        """
        unshuffled_data = list(
            self.data["train"]["examples"],
        )
        random.shuffle(unshuffled_data)
        self.data["train"]["examples"] = list(self.data["train"]["examples"])
