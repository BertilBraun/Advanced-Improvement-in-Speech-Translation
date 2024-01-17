import csv
import os
from collections import defaultdict


def read_data_table():
    # Access the COVOST_ROOT environment variable
    covost_root = os.environ.get("COVOST_ROOT")
    if not covost_root:
        raise EnvironmentError("COVOST_ROOT environment variable is not set")

    # Construct the file path
    file_paths = {
        "train": os.path.join(covost_root, "covost_v2.en_de.train.tsv"),
        "test": os.path.join(covost_root, "covost_v2.en_de.test.tsv"),
        "dev": os.path.join(covost_root, "covost_v2.en_de.dev.tsv"),
    }

    data = defaultdict(list)

    # Read and parse each TSV file
    for split, file_path in file_paths.items():
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                delimiter="\t",
                fieldnames=[
                    "audio_path",
                    "english_sentence",
                    "german_translation",
                    "client_id",
                ],
            )
            for row in reader:
                data[split].append(row)

    return data


if __name__ == "__main__":
    counter = 3
    for split, d in read_data_table().items():
        print(split)
        for i in range(counter):
            print(d[i])
