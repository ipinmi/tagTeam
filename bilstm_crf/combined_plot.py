# @Author: Chibundum Adebayo
import matplotlib.pyplot as plt
import argparse

# Sample usage: python combined_plot.py --lang_code=af --model_path=model

# Arguments for plotting
parser = argparse.ArgumentParser(description="Part of Speech Tagging")
parser.add_argument(
    "--lang_code",
    type=str,
    default="de",
    help="Language code (de/zh/af). Trained on German by default.",
    required=True,
)
parser.add_argument(
    "--model_path",
    help="filepath to save model and evaluation results.",
    default="model",
)

args = parser.parse_args()
lang_code = args.lang_code
model_path = args.model_path


def parse_model_data(file_path):
    """
    Parses the model data from a text file and returns it in a dictionary format.

    Parameters:
    file_path (str): The path to the file containing the model data.

    Returns:
    models (dict): A dictionary containing the the model name as the key and the evaluation data dictionary as the value.
    """
    models = {}

    # Read the loss and F1 scores from the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Keep track of the current model name
    current_model = None
    for line in lines:
        if line.startswith("Model:"):
            current_model = line.split()[1]
            # Initialize the model dictionary and create empty lists for the evaluation metrics
            models[current_model] = {
                "train_loss": [],
                "dev_loss": [],
                "test_loss": [],
                "train_f1": [],
                "dev_f1": [],
                "test_f1": [],
            }
        elif line.startswith("Train Loss:"):
            models[current_model]["train_loss"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )[1:]

        elif line.startswith("Dev Loss:"):
            models[current_model]["dev_loss"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )
        elif line.startswith("Test Loss:"):
            models[current_model]["test_loss"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )
        elif line.startswith("Train F1 Score:"):
            models[current_model]["train_f1"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )
        elif line.startswith("Dev F1 Score:"):
            models[current_model]["dev_f1"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )
        elif line.startswith("Test F1 Score:"):
            models[current_model]["test_f1"] = list(
                map(
                    lambda x: round(float(x), 5),
                    line.split(":")[1].strip()[1:-1].split(", "),
                )
            )

    return models


def plot_model_performance(model_dict):
    """
    Plots the train loss, test loss, dev loss, and F1 scores for each model type.

    Parameters:
    model_dict (dict): A dictionary containing the model name as the key and the evaluation data dictionary as the value.

    Returns:
    The plots of the train loss, test loss, dev loss, and F1 scores for each model type.
    """
    n_epochs = 10
    epochs = range(1, n_epochs + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Ensemble Model Performance")

    markers = [
        "o",
        "*",
        "X",
        "^",
        "s",
        ".",
    ]  # List of markers for each model type
    for idx, (model, data) in enumerate(model_dict.items()):
        # Select the marker for the current model type from the list of markers
        marker = markers[idx]

        axs[0, 0].plot(epochs, data["train_loss"], label=f"{model}", marker=marker)
        axs[0, 1].plot(epochs, data["test_loss"], label=f"{model}", marker=marker)
        axs[1, 0].plot(epochs, data["train_f1"], label=f"{model}", marker=marker)
        axs[1, 1].plot(epochs, data["test_f1"], label=f"{model}", marker=marker)

    axs[0, 0].set_title("Train Loss")
    axs[0, 1].set_title("Test Loss")
    axs[1, 0].set_title("Train F1 Score")
    axs[1, 1].set_title("Test F1 Score")

    for ax in axs.flat:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{model_path}/{lang_code}/{lang_code}-ensemble_plot.png")
    plt.show()


if __name__ == "__main__":
    model_data = parse_model_data(f"{model_path}/{lang_code}/loss_f1_{lang_code}.txt")
    plot_model_performance(model_data)
