# @Author: Chibundum Adebayo

# Sample usage: python train.py --lang_code=zh --data_dir=ud_pos_data --embedding_dir=embeddings --model_path=model
import sys
import argparse
import os
import random
import subprocess
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
from torch.autograd import Variable


from model import POS_Tagger
from data import read_data_file, extract_tokens_tags, get_pretrained_matrix, run_once
from build_vocab import POS_Dataset

from model_params import Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

####################################################################################################################################
# Arguments for training and evaluation
parser = argparse.ArgumentParser(description="Part of Speech Tagging")
parser.add_argument(
    "--lang_code",
    type=str,
    default="de",
    help="Language code (de/zh/af). Trained on German by default.",
    required=True,
)

parser.add_argument(
    "--data_dir",
    help="Directory path for train/dev/test sets.",
    default="ud_pos_data",
    required=True,
)
parser.add_argument(
    "--embedding_dir",
    help="Directory path where the pretrained Fasttext embeddings are stored.",
    default="embeddings",
    required=True,
)
parser.add_argument(
    "--model_path",
    help="filepath to save model and evaluation results.",
    default="model",
)

args = parser.parse_args()
lang_code = args.lang_code
data_dir = args.data_dir
embedding_dir = args.embedding_dir
model_path = args.model_path

subprocess.run(["mkdir", "-p", model_path])
subprocess.run(["mkdir", "-p", f"{model_path}/{lang_code}"])

data_path = f"{data_dir}/{lang_code}-data"
fasttext_model_path = f"{embedding_dir}/cc.{lang_code}.100.bin"
pretrained_weights_path = f"{embedding_dir}/{lang_code}_pretrained_weights.pt"


####################################################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def extract_data(directory):
    """
    Given a directory, this function extracts the paths for the train, dev, and test files.

    Parameters:
    directory: str
        The directory containing the data files

    Returns:
    paths: dict
        A dictionary containing the paths for the train, dev, and test files
    """
    splits = ["train", "dev", "test"]
    paths = {}
    for file in os.listdir(directory):
        for split in splits:
            if split in file and file.endswith(".txt"):
                paths[split] = os.path.join(directory, file)

    return paths


def evalution_with_f1_loss(model, data_split, split_name, best_F=0):
    """
    This function evaluates the model on the given data split
    and returns the micro F1 score and the average loss.

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
    total_loss = 0.0
    total_steps = 0
    expected_tag_seqs = []
    predicted_tag_seqs = []

    model.eval()

    with torch.no_grad():
        for example in data_split:
            # Extend the expected tag sequences for the F1 score calculation
            expected_tag_seqs.extend(example["tag_seq_array"])

            # Word level Input Sequence
            input_sequence = Variable(torch.LongTensor(example["word_array"]))
            input_length = example["word_length"]

            # Character level Input Sequence
            char_input_sequence = example["char_input_list"]

            # Gold standard Tag Sequence
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

            # Convert the padded character sequence to a tensor arranged by length
            paddded_char_seq = np.zeros((len(chars_sorted), char_max_len), dtype="int")

            for i, char in enumerate(chars_sorted):
                paddded_char_seq[i, : char_lengths[i]] = char

            paddded_char_seq = Variable(torch.LongTensor(paddded_char_seq))

            # Get the predicted tags
            _, pred_tags = model(
                input_sequence,
                input_length,
                paddded_char_seq,
                char_lengths,
                char_seq_recover,
                tag_sequence,
            )
            # Calculate the loss
            nll_loss = model.loss_function(
                input_sequence,
                input_length,
                paddded_char_seq,
                char_lengths,
                char_seq_recover,
                tag_sequence,
            )
            if model_params.with_crf:
                pred_tags = pred_tags[0]
                total_loss += nll_loss.detach().numpy() / len(example["word_array"])
            else:
                total_loss += nll_loss.data.item() / len(example["word_array"])

            predicted_tag_seqs.extend(pred_tags)

            total_steps += 1

        avg_loss = total_loss / total_steps

        new_F_score = f1_score(expected_tag_seqs, predicted_tag_seqs, average="micro")

        print()
        if new_F_score > best_F:
            best_F = new_F_score
            print(f"current {split_name} best F_1 Score is {round(best_F, 3)} \n\n")
        return best_F, new_F_score, avg_loss


def train():
    """
    This function trains the model on the training data and returns the epoch loss

    Returns:
        epoch_loss: float
            The average loss over the epoch
        epoch_steps: int
            The number of steps taken in the epoch
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_params.learning_rate,
        momentum=model_params.momentum,
    )
    sys.stdout.flush()

    model.train(True)

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
        chars_sorted = sorted(char_input_sequence, key=lambda p: len(p), reverse=True)

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

        # Adjust the learning rate based on the number of steps
        if epoch_steps % len(final_train_data) == 0:
            adjusted_lr = model_params.learning_rate / (
                1 + 0.05 * epoch_steps / len(final_train_data)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr

    return epoch_loss, epoch_steps


def run_epoch():
    """
    This function runs the training and evaluation loop for the specified number of epochs
    and saves the model with the best validation loss.

    Returns:
        train_loss: list
            The training loss for each epoch
        dev_loss: list
            The validation loss for each epoch
        test_loss: list
            The test loss for each epoch
        train_f1_scores: list
            The training F1 score for each epoch
        dev_f1_scores: list
            The validation F1 score for each epoch
        test_f1_scores: list
            The test F1 score for each epoch
    """
    save_path = f"{model_path}/{lang_code}/{model_name}.pt"

    best_dev_loss = float("inf")
    best_train_Fscore = -1.0
    best_dev_Fscore = -1.0
    best_test_Fscore = -1.0

    train_f1_scores = []
    dev_f1_scores = []
    test_f1_scores = []

    train_loss = []
    dev_loss = []
    test_loss = []
    sys.stdout.flush()

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        # Train the model
        train_epoch_loss, epoch_steps = train()

        # Average the loss over the epoch
        train_epoch_loss /= epoch_steps
        if train_loss == []:
            train_loss.append(train_epoch_loss)
        train_loss.append(train_epoch_loss)

        # Train set F1 score
        best_train_Fscore, epoch_train_Fscore, _ = evalution_with_f1_loss(
            model, final_train_data, "train", best_train_Fscore
        )

        # Validation set F1 score
        best_dev_Fscore, epoch_dev_Fscore, dev_epoch_loss = evalution_with_f1_loss(
            model, final_dev_data, "dev", best_dev_Fscore
        )
        # Test set F1 score
        best_test_Fscore, epoch_test_score, test_epoch_loss = evalution_with_f1_loss(
            model, final_test_data, "test", best_test_Fscore
        )

        sys.stdout.flush()

        train_f1_scores.append(epoch_train_Fscore)
        dev_f1_scores.append(epoch_dev_Fscore)
        test_f1_scores.append(epoch_test_score)

        dev_loss.append(dev_epoch_loss)
        test_loss.append(test_epoch_loss)

        epoch_steps += 1
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if dev_epoch_loss < best_dev_loss:
            best_dev_loss = dev_epoch_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"Train Loss: {train_epoch_loss:.4f} | Train F_1: {epoch_train_Fscore:.4f}"
        )
        print(
            f"Validation Loss: {dev_epoch_loss:.4f} | Validation F_1: {epoch_dev_Fscore:.4f}"
        )
        print(f"Test Loss: {test_epoch_loss:.4f} | Test F_1: {epoch_test_score:.4f}")

    return (
        train_loss,
        dev_loss,
        test_loss,
        train_f1_scores,
        dev_f1_scores,
        test_f1_scores,
    )


def run_and_plot():
    """
    This function collects the evaluation results and plots the evaluation metrics.
    The evaluation results are saved to a text file and the plots are saved as a png file.
    """

    # Train and Evaluate the model
    (
        train_losses,
        dev_losses,
        test_losses,
        train_f1_scores,
        dev_f1_scores,
        test_f1_scores,
    ) = run_epoch()

    # Save the evaluation results
    with open(
        f"{model_path}/{lang_code}/evaluation_results_{model_name}.txt", "w"
    ) as f:
        f.write(f"Train Loss: {train_losses[-1]}\n")
        f.write(f"Dev Loss: {dev_losses[-1]}\n")
        f.write(f"Test Loss: {test_losses[-1]}\n")

        f.write(f"Train F1 Score: {train_f1_scores[-1]}\n")
        f.write(f"Dev F1 Score: {dev_f1_scores[-1]}\n")
        f.write(f"Test F1 Score: {test_f1_scores[-1]}\n")

    # Plot the losses with epochs on the x-axis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        range(0, n_epochs + 1),
        train_losses,
        color="k",
        linestyle="-",
        label="Train",
    )
    plt.plot(
        range(1, n_epochs + 1),
        dev_losses,
        color="b",
        linestyle=":",
        label="Dev",
    )
    plt.plot(
        range(1, n_epochs + 1),
        test_losses,
        color="g",
        linestyle="--",
        label="Test",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Negative Log Likelihood Loss per Epoch (Train, Dev, Test)")
    plt.grid(True)

    # Plot the F1 scores with epochs on the x-axis
    plt.subplot(1, 2, 2)

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
    plt.savefig(f"{model_path}/{lang_code}/evaluation_plots_{model_name}.png")


if __name__ == "__main__":

    # Exract file path for train, test, dev from directory
    data_paths = extract_data(data_path)

    raw_train_data = read_data_file(data_paths["train"])
    raw_dev_data = read_data_file(data_paths["dev"])
    raw_test_data = read_data_file(data_paths["test"])

    _, _, training_data, _ = extract_tokens_tags(raw_train_data)
    _, _, dev_data, _ = extract_tokens_tags(raw_dev_data)
    _, _, test_data, _ = extract_tokens_tags(raw_test_data)

    train_dev_test = POS_Dataset(training_data, test_data, dev_data)

    text_vocab_obj = train_dev_test.get_vocabulary("word")
    char_vocab_obj = train_dev_test.get_vocabulary("char")
    tag_vocab_obj = train_dev_test.get_vocabulary("tags")

    # Create and load the pretrained tensor
    model_params = Parameters(
        text_vocab_obj, char_vocab_obj, tag_vocab_obj, pretrained_weights_path
    )
    n_epochs = model_params.num_epochs

    print(f"Copying pretrained weights of {lang_code} to numpy tensor")

    run_once(
        get_pretrained_matrix(
            model_params, text_vocab_obj, fasttext_model_path, pretrained_weights_path
        )
    )

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

    print(f"Creating a model with the following configuration {model_name}")
    model = POS_Tagger(model_params).to(device=device)

    print(model)

    run_and_plot()
