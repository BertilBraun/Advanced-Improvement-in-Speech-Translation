from typing import Annotated, Dict, List

import torchaudio
from examples.speech_to_text.data_utils import create_zip, extract_fbank_features

from preprocess.process_audio import (
    OVERWRITE_ZIP,
)
from preprocess.process_audio_covost_wav2vec import (
    iterate_over_dataset_range,
    covost_data,
    COVOST_CORPUS_EN_CLIPS,
    read_data_table,
)
from utils import get_logger

logger = get_logger("ProcessAudioCovostMel")

MEL_ROOT = covost_data / "mel"
MEL_ROOT.mkdir(parents=True, exist_ok=True)


def process_dataset_to_wav2vec_embeddings(
    dataset: Dict[str, List[dict]],
    data_split: Annotated[str, "['test', 'train', 'dev']"],
) -> None:
    logger.info(f"Start processing {data_split} data for mel features")

    for _data in iterate_over_dataset_range(dataset.get(data_split)):
        file_name = _data.get("file_name")
        logger.debug(f"File name: {file_name}")
        input_mp3_file_path = COVOST_CORPUS_EN_CLIPS / file_name
        logger.debug(
            f"Processing {input_mp3_file_path}. Type: {type(input_mp3_file_path)}"
        )
        output_file_path = MEL_ROOT / str(file_name).replace(".mp3", ".npy")
        logger.debug(f"Processing {output_file_path}. Type: {type(output_file_path)}")

        if not output_file_path.is_file():
            waveform, sample_rate = torchaudio.load(input_mp3_file_path)
            extract_fbank_features(waveform, sample_rate, output_file_path)

    logger.info(f"Finished processing wav2vec embeddings for {len(dataset)} samples.")


def main():
    dataset = read_data_table()

    encodings_folder = MEL_ROOT / "encodings"
    zip_file = MEL_ROOT / "encodings.zip"
    if not zip_file.is_file() or OVERWRITE_ZIP:
        process_dataset_to_wav2vec_embeddings(dataset, "test")
        create_zip(encodings_folder, zip_file)


if __name__ == "__main__":
    logger.info("Started Processing audio covost: Mel")
    main()
    logger.info("Finished Processing audio covost: Mel")
