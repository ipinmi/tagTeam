# @Author: Chibundum Adebayo

import torch
import random
from collections import defaultdict
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################################################################################


class Vocabulary(object):
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        """
        <pad> and <unk> tokens are set to index 0 and 1 in all vocabs
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.idx2token = [pad_token, unk_token]
        self.token2idx = defaultdict(lambda: self.idx2token.index(self.pad_token))
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


class TaggingDataset(object):
    """
    A class to hold data pairs of input words, characters, and target tags.
    """

    def __init__(
        self,
        train_data: List[Tuple[List, List]],
        test_data: List[Tuple[List, List]],
        dev_data: List[Tuple[List, List]],
    ):
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

        self.add_test_dev_set(test_data, "test")
        self.add_test_dev_set(dev_data, "dev")

    def read_dataset(
        self, input_data: List[Tuple[List, List]], add_to_vocabularies: bool
    ):
        # Convert each example to a tensor and save the length
        examples = []
        for input_list, target_list in input_data:
            assert len(input_list) == len(target_list), "Invalid data example."

            # If false, do not add test words to the vocabularies.
            if add_to_vocabularies:
                self.word_vocab.add_token_sequence(input_list)
                self.tags_vocab.add_token_sequence(target_list)

            # Convert the input sequence to an array of ints.
            input_array = self.sentence_to_array(input_list, vocabulary="word")

            # Convert each word in the sentence into a sequence of ints.
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
        if split == "test":
            test_examples = self.read_dataset(data, add_to_vocabularies=False)
            self.data["test"]["examples"] = test_examples

        elif split == "dev":
            dev_examples = self.read_dataset(data, add_to_vocabularies=False)
            self.data["dev"]["examples"] = dev_examples

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "word":
            vocab = self.word_vocab
        elif vocabulary == "char":
            vocab = self.char_vocab
        elif vocabulary == "tags":
            vocab = self.tags_vocab
        else:
            raise ValueError(
                "Specified unknown vocabulary in sentence_to_array: {}".format(
                    vocabulary
                )
            )
        return vocab

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each str word in a sentence to an integer from the vocabulary.
        :param sentence: the sentence in words (strings).
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = []
        for word in sentence:
            sentence_array.append(vocab.token_to_idx(word))
        return sentence_array

    def get_example(self, idx: int, split="train"):
        if idx >= len(self.data[split]["examples"]):
            raise ValueError(
                "Dataset has no example at idx %d for split %s" % (idx, split)
            )
        word_tensor = self.array_to_sentence(
            self.data[split]["examples"][idx]["word_tensor"], "word"
        )

        # Convert each word in the sentence into an array of integers per char.
        char_inputs = [
            self.array_to_sentence(char_input, "char")
            for char_input in self.data[split]["examples"][idx]["char_input_tensor"]
        ]
        tag_seq_tensor = self.array_to_sentence(
            self.data[split]["examples"][idx]["tag_seq_tensor"], "tags"
        )
        return word_tensor, char_inputs, tag_seq_tensor

    def array_to_sentence(
        self, sentence_array: List[int], vocabulary: str
    ) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [
            vocab.idx_to_token(token_idx) for token_idx in sentence_array.squeeze(dim=0)
        ]

    def shuffle_train_data(self):
        zipped_data = list(
            zip(
                self.data["train"]["examples"],
                self.data["train"]["example_lengths"],
                self.data["train"]["char_max_lengths"],
            )
        )
        random.shuffle(zipped_data)
        (
            self.data["train"]["examples"],
            self.data["train"]["example_lengths"],
            self.data["train"]["char_max_lengths"],
        ) = zip(*zipped_data)
        self.data["train"]["examples"] = list(self.data["train"]["examples"])
        self.data["train"]["example_lengths"] = list(
            self.data["train"]["example_lengths"]
        )
        self.data["train"]["char_max_lengths"] = list(
            self.data["train"]["char_max_lengths"]
        )
