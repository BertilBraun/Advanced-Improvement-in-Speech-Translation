import csv
import os
from pathlib import Path

from pydub import AudioSegment

from preprocess.utils import read_data_table
from utils import get_logger

logger = get_logger("EvalDataPrepareCovost")

# Access the COVOST_CORPUS environment variable
COVOST_DATA = Path(os.environ.get("COVOST_CORPUS"))
if not COVOST_DATA:
    raise EnvironmentError("COVOST_CORPUS environment variable is not set")

# Access the COVOST_CORPUS environment variable
COVOST_CORPUS = Path(os.environ.get("COVOST_CORPUS"))
if not COVOST_CORPUS:
    raise EnvironmentError("COVOST_CORPUS environment variable is not set")


def calculate_number_of_frames(audio_file: str) -> int:
    logger.info(f"Reading audio file {audio_file}")
    # Load the audio file
    audio = AudioSegment.from_mp3(audio_file)

    # Duration in milliseconds and converting it to seconds
    duration_seconds = len(audio) / 1000.0

    # Sample rate (e.g., 44100 for 44.1 kHz)
    sample_rate = audio.frame_rate
    logger.info("Sample rate %s" % sample_rate)

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
    logger.info("Writing to file %s" % output_file_path)
    if not data_list:
        raise ValueError("The data list is empty")

    if headers is None:
        headers = data_list[0].keys()

    with open(output_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(data_list)


if __name__ == "__main__":
    logger.info("Starting preparation")
    out_path = COVOST_DATA / "asr_input_data_tables"
    os.makedirs(out_path, exist_ok=True)

    dataset_tables = read_data_table()

    uid = 0
    for data_split, table in dataset_tables.items():
        logger.info(f"Processing {data_split} ...")
        new_data = []
        for entry in table:
            file_path = COVOST_CORPUS / "en" / "clips" / entry["file_name"]
            row = {
                "id": uid,
                "audio": file_path,
                "n_frames": calculate_number_of_frames(file_path),
                "tgt_text": entry["en"],
                "speaker": entry["client_id"],
            }
            new_data.append(row)
            uid += 1

        write_list_to_tsv(
            data_list=new_data,
            output_file_path=out_path / f"{data_split}.tsv",
            headers=["id", "audio", "n_frames", "tgt_text", "speaker"],
        )
