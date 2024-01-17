import csv
import os
from collections import defaultdict
from pathlib import Path

from pydub import AudioSegment

from preprocess.process_audio_covost_wav2vec import (
    COVOST_CORPUS_EN_CLIPS,
    read_data_table,
)
from utils import get_logger

logger = get_logger("EvalDataPrepareCovost")

COVOST_DATA = Path("/pfs/work7/workspace/scratch/ubppd-ASR/covost/data")
COVOST_CORPUS_EN_CLIPS


def calculate_number_of_frames(audio_file: str) -> int:
    # Load the audio file
    audio = AudioSegment.from_mp3("file.mp3")

    # Duration in milliseconds and converting it to seconds
    duration_seconds = len(audio) / 1000.0

    # Sample rate (e.g., 44100 for 44.1 kHz)
    sample_rate = audio.frame_rate

    # Calculate the total number of frames
    return duration_seconds * sample_rate


def write_list_to_tsv(data_list, output_file_path, headers=None):
    """
    Writes a list of dictionaries to a TSV file.

    :param data_list: List of dictionaries to write to the file.
    :param output_file_path: Path to the output TSV file.
    :param headers: Optional. List of headers for the TSV file. If not provided,
                    the keys of the first dictionary in the list will be used.
    """
    if not data_list:
        raise ValueError("The data list is empty")

    if headers is None:
        headers = data_list[0].keys()

    with open(output_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(data_list)


if __name__ == "__main__":
    out_path = Path(COVOST_DATA) / "asr_input_data_tables"
    os.makedirs(out_path, exist_ok=True)

    dataset_tables = read_data_table()
    fieldnames = [
        "file_name",
        "en",
        "de",
        "client_id",
    ]
    id = 0
    for data_split, table in dataset_tables.items():
        new_data = defaultdict(list)
        for entry in table:
            new_entry = {"id": id}
            id += 1

            file_path = COVOST_DATA / "wav2vec" / entry["file_name"]
            new_entry["audio"] = file_path
            new_entry["n_frames"] = calculate_number_of_frames(file_path)
            new_entry["tgt_text"] = entry["en"]
            new_entry["speaker"] = entry["client_id"]
            new_data[data_split].append(new_entry)
