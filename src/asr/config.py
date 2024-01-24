import pandas as pd

from pathlib import Path
from typing import Literal

from examples.speech_to_text.data_utils import create_zip, gen_config_yaml, gen_vocab, get_zip_manifest, save_df_to_tsv

from src.datasets.st_dataset import STDataset
from src.datasets.util import iterate_over_dataset
from src.logger_utils import get_logger

logger = get_logger("ASR::Config")


def create_asr_configs(
    datasets: list[STDataset],
    dataset_names: list[str],
    root_location: Path,
    encodings_folder: Path,
    zip_file: Path,
    vocab_size: int = 5000,
    spm_filename: str | None = None,
    overwrite_zip: bool = False,
) -> None:
    """Creates the ASR configs for the given datasets and saves them to the given root location.
    If the zip file does not exist or overwrite_zip is True, the zip file will be created as well.
    This function assumes that the datasets are already processed to wav2vec embeddings or mel spectrograms in the given encodings folder.
    Assumes that the first dataset is the training dataset."""

    if not zip_file.is_file() or overwrite_zip:
        create_zip(encodings_folder, zip_file)

    __cleanup_config_folder(root_location)

    for dataset, dataset_name in zip(datasets, dataset_names):
        __process_dataset_manifest(dataset, dataset_name, root_location, zip_file)

    if spm_filename is None or not (root_location / spm_filename).is_file():
        spm_filename = spm_filename or f"spm_unigram{vocab_size}.model"

        __process_dataset_vocab(root_location, datasets[0], spm_filename, vocab_size)

    __process_dataset_config(root_location, spm_filename)


def __cleanup_config_folder(root_location: Path) -> None:
    # delete old tsv, txt, model files from root_location
    for file in root_location.glob("*"):
        if file.is_file() and file.suffix in [".tsv", ".txt", ".model"]:
            file.unlink()


def __process_dataset_manifest(dataset: STDataset, dataset_name: str, root_location: Path, zip_file: Path) -> None:
    def _cleanup_utterance(utt: str) -> str:
        # cleanup the utterance to make it easier for the ASR model to learn
        modified = utt.lower()

        # remove any characters that are not in the alphabet (a-z) or a space or number (0-9)
        modified = "".join(c for c in modified if c.isalpha() or c == " " or c.isdigit())

        return modified

    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

    logger.info("Fetching audio manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_file)

    logger.info(f"Fetching manifest from {dataset_name}...")
    manifest = {c: [] for c in MANIFEST_COLUMNS}

    for path, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset, desc=f"Manifest {dataset_name}"):
        identifier = f"{speaker_id}-{sample_id}"
        manifest["id"].append(identifier)
        manifest["audio"].append(audio_paths[identifier])
        manifest["n_frames"].append(audio_lengths[identifier])
        manifest["tgt_text"].append(_cleanup_utterance(sentence))
        manifest["speaker"].append(speaker_id)

    save_df_to_tsv(
        pd.DataFrame.from_dict(manifest), root_location / f"{dataset_name}.tsv"
    )


def __process_dataset_vocab(root_location: Path, dataset: STDataset, spm_filename: str, vocab_size: int = 5000) -> None:
    # Collect train text to generate sentencepiece model and vocabulary later on
    with open(root_location / "train_text.txt", "w") as f:
        for path, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset, desc=f"Collections train text"):
            f.write(sentence + "\n")

    gen_vocab(
        root_location / "train_text.txt",
        root_location / spm_filename,
        model_type="unigram",
        vocab_size=vocab_size,
    )


def __process_dataset_config(root_location: Path, spm_filename: str, type: Literal["wav2vec"] | Literal["mel"] = "mel") -> None:
    gen_config_yaml(
        root_location, 
        spm_filename=spm_filename,
        input_feat_per_channel=80 if type == "mel" else 768,
    )
