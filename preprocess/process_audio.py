import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from examples.speech_to_text.data_utils import (
    create_zip,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
    extract_fbank_features,
)
from torch.nn.utils.rnn import pad_sequence
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

OVERWRITE_ZIP = True

ROOT_LOCATION = Path(f"{os.environ['HOME']}/ASR")
WORKSPACE_ROOT_LOCATION = Path("/pfs/work7/workspace/scratch/uxude-ASR")

DATASET_LOCATION = ROOT_LOCATION / "data"
DATASET_LOCATION.mkdir(parents=True, exist_ok=True)

WAV2VEC_ROOT = ROOT_LOCATION / "wav2vec"
WAV2VEC_ROOT.mkdir(parents=True, exist_ok=True)

WAV2VEC_WORKSPACE_ROOT = WORKSPACE_ROOT_LOCATION / "wav2vec"
WAV2VEC_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

MEL_ROOT = ROOT_LOCATION / "mel"
MEL_ROOT.mkdir(parents=True, exist_ok=True)

MEL_WORKSPACE_ROOT = WORKSPACE_ROOT_LOCATION / "mel"
MEL_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

ENCODING_FOLDER_NAME = "encoded"
ZIP_FILE_NAME = "encoded.zip"

WAV2VEC_ENCODING_ROOT = WAV2VEC_WORKSPACE_ROOT / ENCODING_FOLDER_NAME
WAV2VEC_ENCODING_ROOT.mkdir(parents=True, exist_ok=True)

MEL_ENCODING_ROOT = MEL_WORKSPACE_ROOT / ENCODING_FOLDER_NAME
MEL_ENCODING_ROOT.mkdir(parents=True, exist_ok=True)

# TODO Define your batch size
BATCH_SIZE = 12

# TODO maybe adjust vocab size - 900 was used in the colab
VOCAB_SIZE = 1000

# TODO For final training set to 'train-clean-360'?
DATASET_NAMES = ["train-clean-360", "dev-clean", "test-clean"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Wav2Vec model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model = model.to(device)


def load_dataset():
    print("Fetching data...")
    return [
        LIBRISPEECH(DATASET_LOCATION.as_posix(), url=dataset_name, download=True)
        for dataset_name in DATASET_NAMES
    ]  # train_data, dev_data, test_data


def for_in_dataset(dataset, desc="", start=0, end=None):
    end = len(dataset) if end is None else end
    for i in tqdm(range(start, end), desc=desc):
        try:
            data = dataset[i]
        except Exception as e:
            print(f"Error loading dataset at {desc} {i}: {e}")
            continue
        yield data


def extract_wav2vec_features_batch(
        waveforms,  #: List[torch.FloatTensor],
        sample_rate,  #: int,
        output_paths,  #: List[Path],
        overwrite=False,  #: bool = False
):
    # Check if all output files exist and if overwrite is not allowed
    if all(path.is_file() for path in output_paths) and not overwrite:
        return

    try:
        # Store original lengths and remove the first dimension
        original_lengths = [wav.shape[1] for wav in waveforms]
        waveforms_squeezed = [wav.squeeze(0) for wav in waveforms]
        batch_waveforms = pad_sequence(waveforms_squeezed, batch_first=True)
        max_padded_length = batch_waveforms.shape[1]

        # Add a channel dimension
        batch_waveforms = batch_waveforms.unsqueeze(1)

        # Process the batch
        processed_values = processor(
            batch_waveforms, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        processed_values = processed_values.squeeze(0).squeeze(1)
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
    except Exception as e:
        print(f"Error at embedding {output_paths}: {e}")
        return


def process_dataset_to_wav2vec_embeddings(dataset):
    batch_waveforms = []
    batch_paths = []

    for (wav, sample_rate, _, spk_id, chapter_no, utt_no) in for_in_dataset(dataset, desc=f"Wav2vec"):
        file = WAV2VEC_ENCODING_ROOT / f"{spk_id}-{chapter_no}-{utt_no}.npy"

        if not file.is_file():
            batch_waveforms.append(wav.to(device=device, dtype=torch.float))
            batch_paths.append(file.as_posix())

        if len(batch_waveforms) == BATCH_SIZE:
            extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)
            batch_waveforms = []
            batch_paths = []

    if batch_waveforms:
        extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)

    print(f"Finished processing wav2vec embeddings for {len(dataset)} samples.")


def process_dataset_to_mel_spectrogram(dataset):
    for (wav, sample_rate, _, spk_id, chapter_no, utt_no) in for_in_dataset(
            dataset, desc=f"Mel"
    ):
        file = MEL_ENCODING_ROOT / f"{spk_id}-{chapter_no}-{utt_no}.npy"
        if not file.is_file():
            extract_fbank_features(wav, sample_rate, file)


def cleanup_utterance(utt: str):
    # cleanup the utterance to make it easier for the ASR model to learn
    modified = utt.lower()

    # remove any characters that are not in the alphabet (a-z) or a space or number (0-9)
    modified = "".join([c for c in modified if c.isalpha() or c == " " or c.isdigit()])

    return modified


def process_dataset_manifest(datasets, root_location, zip_file):
    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

    print("Fetching audio manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_file)

    # assert len(audio_paths) == len(audio_lengths)
    assert len(datasets) == len(DATASET_NAMES)

    text_mapping = {"original": [], "target": []}

    for dataset, dataset_name in zip(datasets, DATASET_NAMES):
        print(f"Fetching manifest from {dataset_name}...")
        manifest = {c: [] for c in MANIFEST_COLUMNS}

        for (_, _, utt, spk_id, chapter_no, utt_no) in for_in_dataset(
                dataset, desc=f"Manifest {dataset_name}"
        ):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(cleanup_utterance(utt))
            manifest["speaker"].append(spk_id)

            text_mapping["original"].append(utt)
            text_mapping["target"].append(cleanup_utterance(utt))

        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), root_location / f"{dataset_name}.tsv"
        )

    save_df_to_tsv(
        pd.DataFrame.from_dict(text_mapping), root_location / "text.tsv"
    )


def process_dataset_vocab(root_location):
    # Collect train text to generate sentencepiece model and vocabulary later on
    train_text = pd.read_csv(root_location / f"{DATASET_NAMES[0]}.tsv", sep="\t")[
        "tgt_text"
    ].tolist()
    with open(root_location / "train_text.txt", "w") as f:
        f.write("\n".join(train_text))

    gen_vocab(
        root_location / "train_text.txt",
        root_location / f"spm_unigram{VOCAB_SIZE}",
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
    )


def process_dataset_config(root_location):
    gen_config_yaml(root_location, spm_filename=f"spm_unigram{VOCAB_SIZE}.model")


# Main execution
def main():
    datasets = load_dataset()  # train_data, dev_data, test_data = datasets

    # Pack audio features into ZIP
    for root_location, workspace_root_location in ((WAV2VEC_ROOT,
                                                    WAV2VEC_WORKSPACE_ROOT),):  # ((WAV2VEC_ROOT, WAV2VEC_WORKSPACE_ROOT), (MEL_ROOT, MEL_WORKSPACE_ROOT)):

        encodings_folder = workspace_root_location / ENCODING_FOLDER_NAME
        zip_file = workspace_root_location / ZIP_FILE_NAME
        if not zip_file.is_file() or OVERWRITE_ZIP:

            for dataset in datasets:
                if root_location == WAV2VEC_ROOT:
                    process_dataset_to_wav2vec_embeddings(dataset)
                if root_location == MEL_ROOT:
                    # Only rank == 0 processes the mel spectrograms as they are not as computationally expensive
                    process_dataset_to_mel_spectrogram(dataset)

            create_zip(encodings_folder, zip_file)

        # delete old tsv, txt, model files from root_location
        for file in root_location.glob("*"):
            if file.is_file() and file.suffix in [".tsv", ".txt", ".model"]:
                file.unlink()

        process_dataset_manifest(datasets, root_location, zip_file)

        process_dataset_vocab(root_location)

        process_dataset_config(root_location)


if __name__ == "__main__":
    main()
