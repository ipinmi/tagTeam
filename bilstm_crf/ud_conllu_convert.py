# @Author: Chibundum Adebayo

import os
import argparse
import subprocess
import shutil

# Sample usage: python ud_conllu_convert.py --lang_code=zh --ud_data_dir=ud_pos_data


def delete_non_conllu_files(directory):
    """
    Function to delete all non-conllu files in the directory

    Parameters:
    directory: str
        The directory for each language UD data

    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # Check if the item is a file and does not end with .conllu
        if os.path.isfile(item_path) and not item.endswith(".conllu"):
            print(f"Deleting file: {item_path}")
            os.remove(item_path)
        # Check if the item is a directory
        elif os.path.isdir(item_path):
            print(f"Deleting folder and its contents: {item_path}")
            shutil.rmtree(item_path)


def extract_pos_files(file_dir, extension="conllu"):
    """
    Extracts all the conllu files from the directory with their full path

    Parameters

    directory: str
        The directory for each language UD data
    extension: str (default="conllu")
        The extension of the files to extract
    """
    dir_files = os.listdir(file_dir)

    conllu_files = []

    for file in dir_files:
        if file.endswith(extension):
            # add the full path to the file
            conllu_files.append(os.path.join(file_dir, file))

    return conllu_files


def convert_ud_data(conllu_filepath):
    """
    Function to convert UD data from conllu format to text format

    CoNLL-U Format Description:
    Each sentence is separated by a new line and each token is written in a line
    and contains several annotation fields separated by a tab character:
        # sent_id
        # text
        ID FORM LEMMATIZED UDPOS XPOS FEAT HEAD DEPREL DEPS MISC

    Target Text Format:
        FORM LEMMATIZED UDPOS
    """
    # initialize the pos data as an empty string
    pos_data = ""
    with open(conllu_filepath, "r") as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith("#"):
                continue
            else:
                pos_data += " ".join(line.split("\t")[1:4]) + "\n"

    # save to a text file
    txt_filepath = conllu_filepath.replace(".conllu", ".txt")
    with open(txt_filepath, "w", encoding="utf-8") as file:
        file.write(pos_data)


def main():
    parser = argparse.ArgumentParser(description="UD CoNLL-U to text format converter")
    parser.add_argument(
        "--lang_code",
        type=str,
        default="de",
        help="Language code (de/zh/af/en). Trained on German by default.",
    )
    parser.add_argument(
        "--ud_data_dir",
        type=str,
        help="Directory where the downloaded UD data is saved. Example: ud_data/UD_German-GSD-master",
    )

    args = parser.parse_args()

    file_dir = args.ud_data_dir
    lang_code = args.lang_code

    download_links = {
        "de": "UD_German-GSD",
        "en": "UD_English-GUM",
        "zh": "UD_Chinese-GSD",
        "af": "UD_Afrikaans-AfriBooms",
    }

    download_link = f"https://github.com/UniversalDependencies/{download_links[lang_code]}/archive/master.zip"
    save_path = f"{file_dir}/master.zip"

    subprocess.run(["mkdir", "-p", file_dir])

    # download the UD POS data
    subprocess.run(
        [
            "curl",
            "-L",
            "-o",
            save_path,
            download_link,
        ]
    )
    # unzip the downloaded UD POS file
    subprocess.run(["unzip", save_path, "-d", f"{file_dir}"])

    # delete the zip file
    subprocess.run(["rm", save_path])

    extract_path = f"{file_dir}/{download_links[lang_code]}-master"
    new_extract_path = f"{file_dir}/{lang_code}-data"

    # Rename the extracted folder to the language code
    subprocess.run(["mv", extract_path, f"{file_dir}/{lang_code}-data"])
    subprocess.run(["rm", extract_path])

    # Delete all non-conllu files
    delete_non_conllu_files(new_extract_path)

    # Extract all the conllu files
    pos_files = extract_pos_files(new_extract_path)

    # Convert the conllu files to text format
    print(f"Found {len(pos_files)} UD CoNLL-U files in the directory")
    for pos_file in pos_files:
        print(f"\tConverting {pos_file} to text format")
        convert_ud_data(pos_file)


if __name__ == "__main__":
    main()
