# @Author: Chibundum Adebayo

import sys
from tqdm import tqdm
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score

from model import POS_Tagger
from data import read_data_file, extract_tokens_tags
from build_vocab import TaggingDataset

from model_params import Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


####################################################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluation(
    model,
    data_split,
    split_name,
    best_F=0,
):
    """
    This funtion evaluates the model on the given data split and returns the micro F1 score

    Parameters:
        model: nn.Module
            The model to be evaluated
        data_split: list
            The data split to be evaluated
        split_name: str
            The name of the data split
        best_F: float (default=0)
            The best F1 score obtained so far
    Returns:
        best_F: float
            The best F1 score obtained so far
        new_F_score: float
            The F1 score obtained on the current data split
    """
    new_F_score = 0.0

    expected_tag_seqs = []
    predicted_tag_seqs = []

    # model.eval()
    with torch.no_grad():
        for example in data_split:
            # extend the expected tag sequence for all the examples
            expected_tag_seqs.extend(example["tag_seq_array"])
            input_sequence = example["word_array"]
            input_sequence = Variable(torch.LongTensor(input_sequence))
            input_length = example["word_length"]
            char_input_sequence = example["char_input_list"]
            tags = example["tag_seq_array"]
            tag_sequence = Variable(torch.LongTensor(tags))

            # Create padded character representation and sort by length
            chars_sorted = sorted(
                char_input_sequence, key=lambda p: len(p), reverse=True
            )

            # Use to recover the char to original sequence positions
            char_seq_recover = {}
            for i, c_i in enumerate(char_input_sequence):
                for j, c_j in enumerate(chars_sorted):
                    if (
                        c_i == c_j
                        and not j in char_seq_recover
                        and not i in char_seq_recover.values()
                    ):
                        char_seq_recover[j] = i
                        continue

            char_lengths = [len(c) for c in chars_sorted]
            char_max_len = max(char_lengths)
            paddded_char_seq = np.zeros((len(chars_sorted), char_max_len), dtype="int")

            for i, char in enumerate(chars_sorted):
                paddded_char_seq[i, : char_lengths[i]] = char

            paddded_char_seq = Variable(torch.LongTensor(paddded_char_seq))

            _, pred_tags = model(
                input_sequence,
                input_length,
                paddded_char_seq,
                char_lengths,
                char_seq_recover,
                tag_sequence,
            )
            if model_params.with_crf:
                pred_tags = pred_tags[0]
            predicted_tag_seqs.extend(pred_tags)

        new_F_score = f1_score(expected_tag_seqs, predicted_tag_seqs, average="micro")

        print()
        if new_F_score > best_F:
            best_F = new_F_score
            print(f"current {split_name} best F_1 Sore is {round(best_F, 3)} \n\n")
        return best_F, new_F_score


def train():
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_params.learning_rate,
        momentum=model_params.momentum,
    )
    n_epochs = model_params.num_epochs
    epoch_losses = []
    epoch_f1_scores = []
    best_dev_Fscore = -1.0
    best_test_Fscore = -1.0
    best_train_Fscore = -1.0
    sys.stdout.flush()

    model.train(True)

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for _, index in enumerate(tqdm(np.random.permutation(len(final_train_data)))):
            data = final_train_data[index]

            # Clear the gradients
            model.zero_grad()

            # Word level Input Sequence
            input_sequence = Variable(torch.LongTensor(data["word_array"]))
            input_length = data["word_length"]

            # Character level Input Sequence
            char_input_sequence = data["char_input_list"]

            # Gold standard Tag Sequence
            tags = data["tag_seq_array"]
            tag_sequence = Variable(torch.LongTensor(tags))

            # Create padded character representation and sort by length
            chars_sorted = sorted(
                char_input_sequence, key=lambda p: len(p), reverse=True
            )

            # Use to recover the char to original sequence positions
            char_seq_recover = {}
            for i, c_i in enumerate(char_input_sequence):
                for j, c_j in enumerate(chars_sorted):
                    if (
                        c_i == c_j
                        and not j in char_seq_recover
                        and not i in char_seq_recover.values()
                    ):
                        char_seq_recover[j] = i
                        continue

            char_lengths = [len(c) for c in chars_sorted]
            char_max_len = max(char_lengths)
            paddded_char_seq = np.zeros((len(chars_sorted), char_max_len), dtype="int")

            for i, char in enumerate(chars_sorted):
                paddded_char_seq[i, : char_lengths[i]] = char

            # Convert the padded character sequence to a tensor arranged by length
            paddded_char_seq = Variable(torch.LongTensor(paddded_char_seq))

            nll_loss = model.loss_function(
                input_sequence,
                input_length,
                paddded_char_seq,
                char_lengths,
                char_seq_recover,
                tag_sequence,
            )
            if model_params.with_crf:
                epoch_loss += nll_loss.detach().numpy() / len(data["word_array"])
            else:
                epoch_loss += nll_loss.data.item() / len(data["word_array"])

            epoch_steps += 1

            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_params.clip)
            optimizer.step()

        # Average the loss over the epoch
        epoch_loss /= epoch_steps
        if epoch_losses == []:
            epoch_losses.append(epoch_loss)
        epoch_losses.append(epoch_loss)

        # Evaluate F1 score on the validation set and test set
        model.eval()

        # Train set F1 score
        best_train_Fscore, epoch_train_Fscore = evaluation(
            model, final_train_data, "train", best_train_Fscore
        )
        # Validation set F1 score
        best_dev_Fscore, epoch_dev_Fscore = evaluation(
            model, final_dev_data, "dev", best_dev_Fscore
        )
        # Test set F1 score
        best_test_Fscore, epoch_test_score = evaluation(
            model, final_test_data, "test", best_test_Fscore
        )
        sys.stdout.flush()

        epoch_f1_scores.append([epoch_train_Fscore, epoch_dev_Fscore, epoch_test_score])

        model.train(True)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(
            f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {epoch_loss:.4f} | Train F_1: {epoch_train_Fscore:.4f}"
        )

    # Plot the losses with epochs on the x-axis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, n_epochs + 1), epoch_losses, marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)

    # Plot the F1 scores with epochs on the x-axis
    plt.subplot(1, 2, 2)

    train_f1_scores = [f1_score[0] for f1_score in epoch_f1_scores]
    dev_f1_scores = [f1_score[1] for f1_score in epoch_f1_scores]
    test_f1_scores = [f1_score[2] for f1_score in epoch_f1_scores]

    plt.plot(
        range(1, n_epochs + 1),
        train_f1_scores,
        color="k",
        linestyle="-",
        label="Train",
    )
    plt.plot(
        range(1, n_epochs + 1),
        dev_f1_scores,
        color="b",
        linestyle=":",
        label="Dev",
    )
    plt.plot(
        range(1, n_epochs + 1),
        test_f1_scores,
        color="g",
        linestyle="--",
        label="Test",
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Epoch (Train, Dev, Test)")
    plt.grid(True)

    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(
        "training_results.png"
    )  # TODO: add dynamic filename based on language and configurations


def perform_eval(model, mode="test"):
    n_epochs = model_params.num_epochs
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(
            f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {epoch_loss:.4f} | Train F_1: {epoch_train_Fscore:.4f}"
        )


if __name__ == "__main__":
    # TODO: Add arguments to the parser for all the hyperparameters
    raw_train_data = read_data_file("../pos_data/ger/de_gsd-ud-train.txt")
    raw_dev_data = read_data_file("../pos_data/ger/de_gsd-ud-dev.txt")
    raw_test_data = read_data_file("../pos_data/ger/de_gsd-ud-test.txt")

    _, _, training_data, _ = extract_tokens_tags(raw_train_data)
    _, _, dev_data, _ = extract_tokens_tags(raw_dev_data)
    _, _, test_data, _ = extract_tokens_tags(raw_test_data)

    train_dev_test = TaggingDataset(training_data, test_data, dev_data)

    text_vocab_obj = train_dev_test.get_vocabulary("word")
    char_vocab_obj = train_dev_test.get_vocabulary("char")
    tag_vocab_obj = train_dev_test.get_vocabulary("tags")

    model_params = Parameters(text_vocab_obj, char_vocab_obj, tag_vocab_obj)

    final_train_data = train_dev_test.data["train"]["examples"]
    final_dev_data = train_dev_test.data["dev"]["examples"]
    final_test_data = train_dev_test.data["test"]["examples"]

    # Model naming convention
    model_name_config = [model_params.lang_code]

    # Add configuration based on model settings
    if model_params.use_char:
        model_name_config.append("char")

    if model_params.use_pretrained:
        model_name_config.append("pretrained")

    model_name_config.append("bilstm")

    if model_params.with_crf:
        model_name_config.append("crf")
    model_name = "_".join(model_name_config)

    model = POS_Tagger(model_params).to(device=device)

    print(model)

    train()
