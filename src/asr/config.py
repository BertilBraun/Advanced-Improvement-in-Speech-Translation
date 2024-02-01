import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable, Literal, Sequence

from examples.speech_to_text.data_utils import gen_config_yaml, gen_vocab, save_df_to_tsv

from src.datasets.base.asr_dataset import ASRDataset
from src.datasets.base.st_dataset import STDataset
from src.datasets.util import iterate_over_dataset, process_dataset_in_parallel
from src.logger_utils import get_logger

logger = get_logger("ASR::Config")


def create_asr_configs(
    datasets: Sequence[STDataset],
    dataset_names: list[str],
    root_location: Path,
    encodings_folder: Path,
    encoding_function: Callable[[ASRDataset, Path], None],
    vocab_size: int = 5000,
    spm_filename: str | None = None,
    overwrite_encodings: bool = False,
) -> None:
    """Creates the ASR configs for the given datasets and saves them to the given root location.
    If the zip file does not exist or overwrite_zip is True, the zip file will be created as well.
    This function assumes that the datasets are already processed to wav2vec embeddings or mel spectrograms in the given encodings folder.
    Assumes that the first dataset is the training dataset."""

    for dataset, dataset_name in zip(datasets, dataset_names):
        if (root_location / f"{dataset_name}.processed_encodings").is_file() and not overwrite_encodings:
            logger.info(f"Encodings for {dataset_name} already exist, skipping encoding creation...")
            continue
        
        logger.info(f"Processing dataset {dataset_name} to encoding...")
        encoding_function(ASRDataset(dataset), encodings_folder)
        
        open(root_location / f"{dataset.split}.processed_encodings", "w").close()
            
    # assuming the folder will only ever be used for one dataset
    # __cleanup_config_folder(root_location)

    __process_dataset_manifests(datasets, dataset_names, root_location, encodings_folder)

    if spm_filename is None or not (root_location / spm_filename).is_file():
        spm_filename = spm_filename or f"spm_unigram{vocab_size}.model"

        __process_dataset_vocab(root_location, datasets[0], spm_filename, vocab_size)

    __process_dataset_config(root_location, spm_filename)


def cleanup_utterance(utt: str) -> str:
    # cleanup the utterance to make it easier for the ASR model to learn
    modified = utt.lower()

    # remove any characters that are not in the alphabet (a-z) or a space or number (0-9)
    modified = "".join(c for c in modified if c.isalpha() or c == " " or c.isdigit())

    return modified


def __cleanup_config_folder(root_location: Path) -> None:
    logger.info("Cleaning up config folder...")
    # delete old tsv, txt, model files from root_location
    for file in root_location.glob("*"):
        if file.is_file() and file.suffix in [".tsv", ".txt", ".model"]:
            file.unlink()
            logger.info(f"Deleted {file}")


def __process_dataset_manifests(datasets: Sequence[STDataset], dataset_names: list[str], root_location: Path, encodings_folder: Path) -> None:
    for dataset, dataset_name in zip(datasets, dataset_names):
        if (root_location / f"{dataset_name}.tsv").is_file():
            logger.info(f"Manifest file for {dataset_name} already exists, skipping manifest creation...")
            continue
        
        manifest = {c: [] for c in ("id", "audio", "n_frames", "tgt_text", "speaker")}

        for path, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset, desc=f"Processing {dataset_name} to manifest"):
            identifier = f"{speaker_id}-{sample_id}"
            
            audio_path = encodings_folder / f"{identifier}.npy"
            
            if not audio_path.is_file():
                logger.warning(f"Missing audio file for {identifier}!")
                return
            
            audio_length = np.load(audio_path).shape[0]
            
            manifest["id"].append(identifier)
            manifest["audio"].append(audio_path.as_posix())
            manifest["n_frames"].append(audio_length)
            manifest["tgt_text"].append(cleanup_utterance(sentence))
            manifest["speaker"].append(speaker_id)

        logger.info(f"Saving manifest for {dataset_name}...")
        save_df_to_tsv(pd.DataFrame.from_dict(manifest), root_location / f"{dataset_name}.tsv")


def __process_dataset_vocab(root_location: Path, dataset: STDataset, spm_filename: str, vocab_size: int = 5000) -> None:
    if (root_location / spm_filename).is_file():
        logger.info("Sentencepiece model already exists, skipping sentencepiece model creation...")
        return
    
    # Collect train text to generate sentencepiece model and vocabulary later on
    logger.info("Collecting train text...")
    with open(root_location / "train_text.txt", "w") as f:
        for path, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset, desc=f"Collections train text"):
            f.write(cleanup_utterance(sentence) + "\n")

    logger.info("Generating sentencepiece model and vocabulary...")
    gen_vocab(
        root_location / "train_text.txt",
        root_location / Path(spm_filename).stem,
        model_type="unigram",
        vocab_size=vocab_size,
    )
    logger.info("Done generating sentencepiece model and vocabulary...")
    
    # remove train text file
    (root_location / "train_text.txt").unlink()


def __process_dataset_config(root_location: Path, spm_filename: str, type: Literal["wav2vec"] | Literal["mel"] = "mel") -> None:
    logger.info("Generating config yaml...")
    gen_config_yaml(
        root_location, 
        spm_filename=spm_filename,
        input_feat_per_channel=80 if type == "mel" else 768,
    )
    logger.info("Generated config yaml...")
