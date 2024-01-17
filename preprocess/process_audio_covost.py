import csv
import os
from collections import defaultdict
from pathlib import Path

from utils import get_logger

logger = get_logger()

# Access the COVOST_ROOT environment variable
covost_root = Path(os.environ.get("COVOST_ROOT"))
if not covost_root:
    raise EnvironmentError("COVOST_ROOT environment variable is not set")

covost_corpus_clips = Path(os.environ.get("COVOST_CORPUS")) / "en/clips"


def read_data_table():
    logger.info("Reading Covost data table")

    # Construct the file path
    file_paths = {
        "train": os.path.join(
            covost_root / "translations", "covost_v2.en_de.train.tsv"
        ),
        "test": os.path.join(covost_root / "translations", "covost_v2.en_de.test.tsv"),
        "dev": os.path.join(covost_root / "translations", "covost_v2.en_de.dev.tsv"),
    }

    data = defaultdict(list)

    # Read and parse each TSV file
    for split, file_path in file_paths.items():
        first_row = True
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                delimiter="\t",
                fieldnames=[
                    "file_name",
                    "en",
                    "de",
                    "client_id",
                ],
            )
            for row in reader:
                if first_row:
                    first_row = False
                    continue
                data[split].append(row)

    return data


if __name__ == "__main__":
    counter = 3
    for split, d in read_data_table().items():
        print(split)
        for i in range(counter):
            print(d[i])
