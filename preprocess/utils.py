import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from utils import get_logger

logger = get_logger("Preprocess::Utils")

# Access the COVOST_ROOT environment variable
covost_data = Path(os.environ.get("COVOST_DATA"))
if not covost_data:
    raise EnvironmentError("COVOST_DATA environment variable is not set")


def read_data_table() -> Dict[str, List[dict]]:
    logger.info("Reading Covost data table")

    # Construct the file path
    file_paths = {
        "train": os.path.join(
            covost_data / "translations", "covost_v2.en_de.train.tsv"
        ),
        "test": os.path.join(covost_data / "translations", "covost_v2.en_de.test.tsv"),
        "dev": os.path.join(covost_data / "translations", "covost_v2.en_de.dev.tsv"),
    }

    data_table = defaultdict(list)

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
                data_table[split].append(row)

    logger.info("Finished Reading Covost data tables ")
    return data_table
