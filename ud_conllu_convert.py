import os
import argparse


# TODO: Add a function to download UD data from github
def download_ud_data():
    """
    Function to download UD data from github for converting to text format

    #curl -L -O https://github.com/UniversalDependencies/UD_German-GSD/archive/master.zip && unzip master.zip
    """
    pass


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ud_data_dir",
        type=str,
        help="Directory where the downloaded UD data is saved. Example: ud_data/UD_German-GSD-master",
    )
    args = parser.parse_args()

    file_dir = args.ud_data_dir

    pos_files = extract_pos_files(file_dir)

    print(f"Found {len(pos_files)} UD CoNLL-U files in the directory")
    for pos_file in pos_files:
        print(f"\tConverting {pos_file} to text format")
        convert_ud_data(pos_file)


if __name__ == "__main__":
    main()
