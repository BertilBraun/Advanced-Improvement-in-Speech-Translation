print("hello")
USE_MPI = False

if USE_MPI:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    rank = 0
    size = 1
    
from curses.ascii import US
import torch
from torchaudio.datasets import LIBRISPEECH
import tqdm
from pathlib import Path

import torch
from pathlib import Path
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import pandas as pd

print("middle import")

from examples.speech_to_text.data_utils import (
    create_zip,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)

from torchaudio.datasets import LIBRISPEECH
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np
import os

print("imports done")

# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/

ROOT_LOCATION = Path(f"{os.getenv('HOME')}/fairseq/examples/speech_to_text")
print(f"{ROOT_LOCATION = }")
# ROOT_LOCATION = Path("~/test")


# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/dataset/
DATASET_LOCATION = ROOT_LOCATION / "data"
# Create folder if not yet exist
DATASET_LOCATION.mkdir(exist_ok=True)

# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/encoded_dataset/
ENCODED_OUTPUT_ROOT = ROOT_LOCATION / "encoded"
# Create folder if not yet exist
ENCODED_OUTPUT_ROOT.mkdir(exist_ok=True)

OUTPUT_EMBEDDING_ZIP = ROOT_LOCATION / "encoded.zip"

# TODO Define your batch size
BATCH_SIZE = 12

# TODO maybe adjust vocab size - 900 was used in the colab
VOCAB_SIZE = 5000

wav2VecDevice = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Wav2Vec model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model = model.to(wav2VecDevice)


def load_dataset() -> tuple[LIBRISPEECH, LIBRISPEECH, LIBRISPEECH]:
    
    print("Fetching training data...")
    train_data = LIBRISPEECH(DATASET_LOCATION.as_posix(), url="train-clean-100", download=True) # TODO For final training set to 'train-clean-360'
    print("Fetching dev data...")
    dev_data = LIBRISPEECH(DATASET_LOCATION.as_posix(), url="dev-clean", download=True)
    print("Fetching test data...")
    test_data = LIBRISPEECH(DATASET_LOCATION.as_posix(), url="test-clean", download=True)
    
    return train_data, dev_data, test_data


def ensure_dataset_loaded() -> tuple[LIBRISPEECH, LIBRISPEECH, LIBRISPEECH]:
    if rank == 0:
        # Only rank == 0 may download, otherwise conflicts will appear        
        datasets = load_dataset()
        
        if USE_MPI:
            comm.Barrier()
        
        return datasets
    
    if USE_MPI:
        comm.Barrier()
    
    return load_dataset()
     

def extract_wav2vec_features_batch(
    waveforms: List[torch.FloatTensor],
    sample_rate: int,
    output_paths: List[Path],
    overwrite: bool = False
) -> None:
    # Check if all output files exist and if overwrite is not allowed
    if all(path.is_file() for path in output_paths) and not overwrite:
        return

    # Store original lengths and remove the first dimension
    original_lengths = [wav.shape[1] for wav in waveforms]
    waveforms_squeezed = [wav.squeeze(0) for wav in waveforms]
    batch_waveforms = pad_sequence(waveforms_squeezed, batch_first=True)
    max_padded_length = batch_waveforms.shape[1]

    # Add a channel dimension
    batch_waveforms = batch_waveforms.unsqueeze(1)

    # Process the batch
    processed_values = processor(batch_waveforms, sampling_rate=sample_rate, return_tensors="pt").input_values
    processed_values = processed_values.squeeze(0).squeeze(1)
    processed_values = processed_values.to(wav2VecDevice)

    with torch.no_grad():
        features = model(processed_values).last_hidden_state
    feature_length = features.shape[1]

    # Trim the features based on original lengths and save
    for feature, original_length, output_path in zip(features, original_lengths, output_paths):
        # Calculate the length ratio and apply it to the feature length
        length_ratio = original_length / max_padded_length
        trimmed_length = int(length_ratio * feature_length)

        # Optionally add a small buffer to avoid cutting off too much
        buffer = 5  # You can adjust this buffer
        trimmed_length = min(trimmed_length + buffer, feature_length)

        # Trim the feature
        trimmed_feature = feature.cpu().numpy()[:trimmed_length, :]
        np.save(output_path, trimmed_feature)


def process_dataset_to_embeddings(dataset) -> None:
    total_data = len(dataset)
    data_per_node = total_data // size

    start = rank * data_per_node
    end = (rank + 1) * data_per_node if rank != size - 1 else total_data

    batch_waveforms = []
    batch_paths = []

    for i in tqdm.tqdm(range(start, end), desc=f"Node {rank}"):
        wav, sample_rate, _, spk_id, chapter_no, utt_no = dataset[i]
        batch_waveforms.append(torch.FloatTensor(wav))
        batch_paths.append(ENCODED_OUTPUT_ROOT / f"{spk_id}-{chapter_no}-{utt_no}.npy")

        if len(batch_waveforms) == BATCH_SIZE:
            extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)
            batch_waveforms = []
            batch_paths = []

    if batch_waveforms:
        extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)

    print(f"Node {rank} finished")


def process_dataset_manifest(train_data, dev_data, test_data):
    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

    print("Fetching audio manifest...")
    audio_paths, audio_lengths = get_zip_manifest(OUTPUT_EMBEDDING_ZIP)

    for dataset, split_name in zip([train_data, dev_data, test_data],
                                ["train-clean-100", "dev-clean", "test-clean"]):
        
        print(f"Fetching manifest from {split_name}...")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        
        for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(audio_paths[sample_id])
            manifest["n_frames"].append(audio_lengths[sample_id])
            manifest["tgt_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
            
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), ROOT_LOCATION / f"{split_name}.tsv"
        )


def process_dataset_vocab():
    # Collect train text to generate sentencepiece model and vocabulary later on
    train_text = pd.read_csv(ROOT_LOCATION / "train-clean-100.tsv", sep='\t')["tgt_text"].tolist()
    with open(ROOT_LOCATION / 'train_text.txt', 'w') as f:
        f.write("\n".join(train_text))
        
    gen_vocab(
        ROOT_LOCATION / 'train_text.txt',
        ROOT_LOCATION / f"spm_unigram{VOCAB_SIZE}",
        model_type='unigram',
        vocab_size=VOCAB_SIZE,
    )
    

def process_dataset_config():
    gen_config_yaml(
        ROOT_LOCATION,
        spm_filename=f"spm_unigram{VOCAB_SIZE}.model"
    )


# Main execution
def main() -> None:    
    print("starting")
    datasets = ensure_dataset_loaded() # train_data, dev_data, test_data = datasets
    print("dataset is set")

    for dataset in datasets:
        process_dataset_to_embeddings(dataset)

    if rank != 0:
        # Only continue with root process, as the following steps are not as computationally expensive
        return
    
    # Pack audio features into ZIP
    print("ZIPing features...")
    create_zip(ENCODED_OUTPUT_ROOT, OUTPUT_EMBEDDING_ZIP)

    process_dataset_manifest(*datasets)
    
    process_dataset_vocab()
    
    process_dataset_config()
    
    
if __name__ == "__main__":
    main()
    
