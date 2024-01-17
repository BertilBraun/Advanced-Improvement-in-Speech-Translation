import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Dict, List

from examples.speech_to_text.data_utils import (
    create_zip,
)

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from preprocess.process_audio import model, device, BATCH_SIZE, processor, OVERWRITE_ZIP
from utils import get_logger

logger = get_logger("ProcessAudioCovost")

# Access the COVOST_ROOT environment variable
covost_data = Path(os.environ.get("COVOST_DATA"))
if not covost_data:
    raise EnvironmentError("COVOST_ROOT environment variable is not set")

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


def extract_wav2vec_features_batch(
    input_paths: List[Path],
    output_paths: List[Path],
    overwrite: bool = False,
) -> None:
    logger.info("Extracting wav2vec")
    # Check if all output files exist and if overwrite is not allowed
    if all(path.is_file() for path in output_paths) and not overwrite:
        return

    try:
        waveforms = []
        sample_rates = []
        original_lengths = []

        # Load waveforms and their respective sample rates
        for file_path in input_paths:
            waveform, sample_rate = torchaudio.load(file_path)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)
            original_lengths.append(waveform.shape[1])

        # Remove the first dimension (channel dimension) if it's 1
        waveforms_squeezed = [
            wav.squeeze(0) if wav.shape[0] == 1 else wav for wav in waveforms
        ]
        batch_waveforms = pad_sequence(waveforms_squeezed, batch_first=True)
        max_padded_length = batch_waveforms.shape[1]

        # Add a channel dimension (if your model expects it, otherwise skip this step)
        batch_waveforms = batch_waveforms.unsqueeze(1)

        # Process the batch
        processed_values = processor(
            batch_waveforms, sampling_rate=sample_rates[0], return_tensors="pt"
        ).input_values
        processed_values = processed_values.squeeze(1)
        processed_values = processed_values.to(device)

        with torch.no_grad():
            features = model(processed_values).last_hidden_state
        feature_length = features.shape[1]

        # Trim the features based on original lengths and save
        for feature, original_length, output_path in zip(
            features, original_lengths, output_paths
        ):
            # Calculate the length ratio and apply it to the feature length
            length_ratio = original_length / max_padded_length
            trimmed_length = int(length_ratio * feature_length)

            # Optionally add a small buffer to avoid cutting off too much
            buffer = 5  # You can adjust this buffer
            trimmed_length = min(trimmed_length + buffer, feature_length)

            # Trim the feature
            trimmed_feature = feature.cpu().numpy()[:trimmed_length, :]
            np.save(output_path, trimmed_feature)

            logger.info("Finished extracting wav2vec features")
    except Exception as e:
        logger.error(f"Error at embedding {output_paths}: {e}")
        return


def process_dataset_to_wav2vec_embeddings(
    dataset: Dict[str, List[dict]],
    data_split: Annotated[str, "['test', 'train', 'dev']"],
) -> None:
    logger.info(f"Start processing {data_split} data for wav2vec embeddings")
    input_paths = []
    batch_paths = []

    for _data in dataset.get(data_split):
        file_name = _data.get("file_name")
        logger.info(f"File name: {file_name}")
        input_mp3_file_path = COVOST_CORPUS_EN_CLIPS / file_name
        logger.info(
            f"Processing {input_mp3_file_path}. Type: {type(input_mp3_file_path)}"
        )
        output_file_path = WAV2VEC_ROOT / str(file_name).replace(".mp3", ".npy")
        logger.info(f"Processing {output_file_path}. Type: {type(output_file_path)}")

        if not output_file_path.is_file():
            input_paths.append(input_mp3_file_path)
            batch_paths.append(output_file_path.as_posix())

        if len(input_paths) == BATCH_SIZE:
            extract_wav2vec_features_batch(input_paths, batch_paths)
            input_paths = []
            batch_paths = []

    if input_paths:
        extract_wav2vec_features_batch(input_paths, batch_paths)

    logger.info(f"Finished processing wav2vec embeddings for {len(dataset)} samples.")


def main():
    dataset = read_data_table()

    encodings_folder = WAV2VEC_ROOT / "encodings"
    zip_file = WAV2VEC_ROOT / "encodings.zip"
    if not zip_file.is_file() or OVERWRITE_ZIP:
        process_dataset_to_wav2vec_embeddings(dataset, "train")
        create_zip(encodings_folder, zip_file)


if __name__ == "__main__":
    logger.info("Started Processing audio covost")
    main()
    logger.info("Finished Processing audio covost")
