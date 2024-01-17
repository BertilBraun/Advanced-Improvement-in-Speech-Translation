import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Dict, List

import torchaudio
import torchaudio.transforms as T
from examples.speech_to_text.data_utils import create_zip
from tqdm import tqdm

from preprocess.process_audio import (
    BATCH_SIZE,
    OVERWRITE_ZIP,
    extract_wav2vec_features_batch,
)
from utils import get_logger

logger = get_logger("ProcessAudioCovost")

# Access the COVOST_ROOT environment variable
covost_data = Path(os.environ.get("COVOST_DATA"))
if not covost_data:
    raise EnvironmentError("COVOST_DATA environment variable is not set")

# Access the COVOST_CORPUS environment variable
COVOST_CORPUS = Path(os.environ.get("COVOST_CORPUS"))
if not covost_data:
    raise EnvironmentError("COVOST_CORPUS environment variable is not set")

COVOST_CORPUS_EN_CLIPS = COVOST_CORPUS / "en" / "clips"
logger.info(
    f"Loading corpus from {COVOST_CORPUS_EN_CLIPS}. File type is {type(COVOST_CORPUS_EN_CLIPS)}"
)

WAV2VEC_ROOT = covost_data / "wav2vec"
WAV2VEC_ROOT.mkdir(parents=True, exist_ok=True)


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


def iterate_over_dataset_range(dataset, desc="", start=0, end=None):
    end = len(dataset) if end is None else end
    for i in tqdm(range(start, end), desc=desc):
        try:
            data = dataset[i]
        except Exception as e:
            logger.error(f"Error loading dataset at {desc} {i}: {e}")
            continue
        yield data


def process_dataset_to_wav2vec_embeddings(
    dataset: Dict[str, List[dict]],
    data_split: Annotated[str, "['test', 'train', 'dev']"],
) -> None:
    logger.info(f"Start processing {data_split} data for wav2vec embeddings")
    waveforms = []
    batch_paths = []
    require_sample_rate = 16000
    # Define a resampler to convert the sampling rate to 16000 Hz
    resampler = T.Resample(orig_freq=48000, new_freq=require_sample_rate)

    for _data in iterate_over_dataset_range(dataset.get(data_split)):
        file_name = _data.get("file_name")
        logger.debug(f"File name: {file_name}")
        input_mp3_file_path = COVOST_CORPUS_EN_CLIPS / file_name
        logger.debug(
            f"Processing {input_mp3_file_path}. Type: {type(input_mp3_file_path)}"
        )
        output_file_path = WAV2VEC_ROOT / str(file_name).replace(".mp3", ".npy")
        logger.debug(f"Processing {output_file_path}. Type: {type(output_file_path)}")

        if not output_file_path.is_file():
            waveform, sample_rate = torchaudio.load(input_mp3_file_path)
            if sample_rate != require_sample_rate:
                waveform = resampler(waveform)
            waveforms.append(waveform)
            batch_paths.append(output_file_path)

        if len(waveforms) == BATCH_SIZE:
            extract_wav2vec_features_batch(
                waveforms=waveforms,
                sample_rate=require_sample_rate,
                output_paths=batch_paths,
            )
            waveforms = []
            batch_paths = []

    if waveforms:
        extract_wav2vec_features_batch(
            waveforms=waveforms,
            sample_rate=require_sample_rate,
            output_paths=batch_paths,
        )

    logger.info(f"Finished processing wav2vec embeddings for {len(dataset)} samples.")


def main():
    dataset = read_data_table()

    encodings_folder = WAV2VEC_ROOT / "encodings"
    zip_file = WAV2VEC_ROOT / "encodings.zip"
    if not zip_file.is_file() or OVERWRITE_ZIP:
        process_dataset_to_wav2vec_embeddings(dataset, "test")
        create_zip(encodings_folder, zip_file)


if __name__ == "__main__":
    logger.info("Started Processing audio covost: Wave2Vec")
    main()
    logger.info("Finished Processing audio covost: Wave2Ves")
