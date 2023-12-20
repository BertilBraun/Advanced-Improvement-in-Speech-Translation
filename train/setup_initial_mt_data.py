
from pathlib import Path
from typing import Literal, Union

import requests
import itertools
import os
import random
import sentencepiece as spm

ROOT_FOLDER = Path(f"{os.environ['HOME']}/MT")

DATASET_FOLDER = ROOT_FOLDER / "dataset"
DATASET_FOLDER.mkdir(parents=True, exist_ok=True)

DATASET_EN = DATASET_FOLDER / "train.de-en.en"
DATASET_DE = DATASET_FOLDER / "train.de-en.de"

DATASET_URL = "https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5"


def ensure_dataset_loaded() -> None:
    if not DATASET_EN.is_file() or not DATASET_DE.is_file():

        download_location = DATASET_FOLDER / "data.zip"
        # Download and unzip command
        download_command = f"wget -nc -O '{download_location}' '{DATASET_URL}'"
        unzip_command = f"unzip -o '{download_location}' -d '{DATASET_FOLDER}'"

        # Execute download and unzip commands
        print("Checking and downloading dataset...")
        os.system(download_command)
        print("Unzipping dataset...")
        os.system(unzip_command)
        print("Dataset ready.")



def main() -> None:
    ensure_dataset_loaded()

    spm.SentencePieceTrainer.train(input=f"{DATASET_DE},{DATASET_EN}",
                                model_prefix="bpe",
                                vocab_size=10000)

    print('Finished training sentencepiece model.')
    
    spm_model = spm.SentencePieceProcessor(model_file="bpe.model")

    for partition in ["train", "dev", "tst"]:
        for lang in ["de", "en"]:
            with open(f"{DATASET_FOLDER}/{partition}.de-en.{lang}", "r") as f_in, \
                 open(f"{DATASET_FOLDER}/spm.{partition}.de-en.{lang}", "w") as f_out:
                for line_idx, line in enumerate(f_in.readlines()):
                    # Segmented into subwords
                    line_segmented = spm_model.encode(line.strip(), out_type=str)
                    # Join the subwords into a string
                    line_segmented = " ".join(line_segmented)
                    f_out.write(line_segmented + "\n")


if __name__ == "__main__":
    main()
