# @Author: Chibundum Adebayo

# Sample usage: python fasttext_embed.py --lang_code=af  --embedding_dir=embeddings

import argparse
import subprocess
import fasttext
import fasttext.util


def load_save_fasttext(embedding_dir, lang_code, word_dim=100):
    """
    This function loads the FastText embeddings for the specified language code and
    reduces the dimensionality to the specified word_dim.
    The reduced embeddings are then saved to a binary file.

    Language code:
        German: de
        Chinese: zh
        Afrikaans: af
        English: en

    Parameters:
    embedding_dir (str): Directory path for storing the pretrained embeddings.
    lang_code (str): Language code for the FastText embeddings.

    Returns:
    The reduced FastText embeddings are saved to the specified directory.
    """

    # Load language specific FastText embeddings
    unzip_download_path = f"{embedding_dir}/cc.{lang_code}.300.bin"
    model_save_path = f"{embedding_dir}/cc.{lang_code}.100.bin"

    embedding = fasttext.load_model(unzip_download_path)

    # Reduce the dimensionality of the embeddings to expected word_dim
    fasttext.util.reduce_model(embedding, word_dim)
    assert embedding.get_dimension() == word_dim

    embedding.save_model(model_save_path)


def main():
    parser = argparse.ArgumentParser(description="Part of Speech Tagging")
    parser.add_argument(
        "--lang_code",
        type=str,
        default="de",
        help="Language code (de/zh/af/en). Trained on German by default.",
    )

    """parser.add_argument(
        "--data_dir", help="Directory path for train/dev/test sets.", required=True
    )"""
    parser.add_argument(
        "--embedding_dir",
        help="Directory path for pretrained Fasttext embeddings.",
        required=True,
    )
    """parser.add_argument(
        "--model_path",
        help="filepath to save model and evaluation results.",
        required=True,
    )"""

    args = parser.parse_args()

    lang_code = args.lang_code
    # data_dir = args.data_dir
    embedding_dir = args.embedding_dir
    # model_path = args.model_path  # TODO: add model name to path

    # Download and load the FastText embeddings

    download_link = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang_code}.300.bin.gz"
    zip_download_path = f"{embedding_dir}/cc.{lang_code}.300.bin.gz"

    # Create the directory for storing the pretrained embeddings if it doesn't exist
    subprocess.run(["mkdir", "-p", embedding_dir])

    # Download the embeddings using curl and save it to the specified directory
    subprocess.run(
        [
            "curl",
            "-o",
            zip_download_path,
            download_link,
        ]
    )

    # Unzip the downloaded file
    subprocess.run(["gunzip", zip_download_path])

    # reduce the dimensionality of the embeddings to 100 and save it
    load_save_fasttext(embedding_dir, lang_code, word_dim=100)

    subprocess.run(["rm", f"{embedding_dir}/cc.{lang_code}.300.bin"])

    # Extract the pretrained weights and save to numpy file
    # get_pretrained_matrix(embedding_dir, lang_code)


if __name__ == "__main__":
    main()
