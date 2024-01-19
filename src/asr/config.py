from pathlib import Path

import pandas as pd
from examples.speech_to_text.data_utils import (  # noqa
    create_zip,  # noqa
    gen_config_yaml,  # noqa
    gen_vocab,  # noqa
    get_zip_manifest,  # noqa
    save_df_to_tsv,  # noqa
)
from torch.utils.data import Dataset

from src.datasets.util import iterate_over_dataset
from src.logger_utils import get_logger

logger = get_logger("ASR::Config")


def create_asr_configs(
    datasets: list[Dataset],
    dataset_names: list[str],
    root_location: Path,
    encodings_folder: Path,
    zip_file: Path,
    vocab_size: int = 5000,
    spm_filename: Path = None,
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

        __process_dataset_vocab(
            root_location, dataset_names[0], vocab_size, spm_filename
        )

    __process_dataset_config(root_location, spm_filename)


def __cleanup_config_folder(root_location: Path) -> None:
    # delete old tsv, txt, model files from root_location
    for file in root_location.glob("*"):
        if file.is_file() and file.suffix in [".tsv", ".txt", ".model"]:
            file.unlink()


def __process_dataset_manifest(
    dataset: Dataset, dataset_name: str, root_location: Path, zip_file: Path
) -> None:
    def _cleanup_utterance(utt: str) -> str:
        # cleanup the utterance to make it easier for the ASR model to learn
        modified = utt.lower()

        # remove any characters that are not in the alphabet (a-z) or a space or number (0-9)
        modified = "".join(
            [c for c in modified if c.isalpha() or c == " " or c.isdigit()]
        )

        return modified

    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

    logger.info("Fetching audio manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_file)

    logger.info(f"Fetching manifest from {dataset_name}...")
    manifest = {c: [] for c in MANIFEST_COLUMNS}

    for _, _, utt, spk_id, chapter_no, utt_no in iterate_over_dataset(
        dataset, desc=f"Manifest {dataset_name}"
    ):
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        manifest["id"].append(sample_id)
        manifest["audio"].append(audio_paths[sample_id])
        manifest["n_frames"].append(audio_lengths[sample_id])
        manifest["tgt_text"].append(_cleanup_utterance(utt))
        manifest["speaker"].append(spk_id)

    save_df_to_tsv(
        pd.DataFrame.from_dict(manifest), root_location / f"{dataset_name}.tsv"
    )


def __process_dataset_vocab(
    root_location, train_dataset_name: str, spm_filename: Path, vocab_size: int = 5000
) -> None:
    # Collect train text to generate sentencepiece model and vocabulary later on
    train_text = pd.read_csv(root_location / f"{train_dataset_name}.tsv", sep="\t")[
        "tgt_text"
    ].tolist()
    with open(root_location / "train_text.txt", "w") as f:
        f.write("\n".join(train_text))

    gen_vocab(
        root_location / "train_text.txt",
        root_location / spm_filename,
        model_type="unigram",
        vocab_size=vocab_size,
    )


def __process_dataset_config(root_location, spm_filename: Path) -> None:
    gen_config_yaml(root_location, spm_filename=spm_filename.as_posix())
