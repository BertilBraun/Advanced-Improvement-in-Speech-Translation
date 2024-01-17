from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)

# from IPython.display import Audio
from pathlib import Path
print(1)
#import shutil
#from tempfile import NamedTemporaryFile
from typing import Optional, Tuple
#import librosa
print(2)

import pandas as pd
print(3)
import torchaudio
print(4)
from torch import Tensor
print(5)
from torch.utils.data import Dataset
print(6)
from tqdm import tqdm
print(7)
import warnings
print(8)

warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", None)

COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost/corpus-16.1"
SOURCE_LANG_ID = "en"
TARGET_LANG_ID = "de"
SAMPLE_RATE = 48000
sample_idx = 10

#train_df = pd.read_csv(f"{COVOST_ROOT}/{SOURCE_LANG_ID}/train.tsv", index_col=0)
#print(train_df[["sentence", "translation"]].iloc[sample_idx])

# Audio(filename=f"{COVOST_ROOT}/{SOURCE_LANG_ID}/clips/{train_df.iloc[sample_idx]['path']}",
#      rate=SAMPLE_RATE)


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).
    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str): target (text) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: str,
    ) -> None:
        self.root = Path(root)

        data = pd.read_csv(self.root / f"{split}.tsv", sep='\t').to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]

        self.data = []
        for e in data:
            path = self.root / "clips" / e["path"]
            self.data.append(e)

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path, sr=None, mono=False)
        #waveform, sample_rate = librosa.load(path, sr=None, mono=False)
        waveform = torch.tensor(waveform)[None, :]
        sentence = data["sentence"]
        translation = data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


# Define file path to store the features
root = Path(COVOST_ROOT).absolute() / SOURCE_LANG_ID
feature_root = root / "fbank80"
feature_root.mkdir(exist_ok=True)

# Extract features of all audios in all data splits
for split in CoVoST.SPLITS:
    print(f"Fetching split {split}...")
    dataset = CoVoST(root, split, SOURCE_LANG_ID, TARGET_LANG_ID)

    print(f"Extracting log mel filter bank features of audio")
    for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
        extract_fbank_features(waveform, sample_rate, feature_root / f"{utt_id}.npy")


# Pack audio features into ZIP
zip_path = root / "fbank80.zip"
print("ZIPing features...")
create_zip(feature_root, zip_path)

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

utt_id = "common_voice_en_100764"
feat = np.load(root / "fbank80" / f"{utt_id}.npy")
print(f'Feature vector shape: {feat.shape}')

# Plot the extracted feature
fig, ax = plt.subplots(figsize=(20, 4))
cax = ax.imshow(np.transpose(feat), interpolation='nearest', cmap=cm.afmhot, origin='lower')
ax.set_title('log mel filter bank features')

#plt.show()
"""

import pandas as pd

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

print("Fetching audio manifest...")
audio_paths, audio_lengths = get_zip_manifest(zip_path)

# Generate TSV manifest
task = f"st_{SOURCE_LANG_ID}_{TARGET_LANG_ID}"

for split in CoVoST.SPLITS:
    print(f"Fetching manifest from {split}...")
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    dataset = CoVoST(root, split, SOURCE_LANG_ID, TARGET_LANG_ID)

    for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
        manifest["id"].append(utt_id)
        manifest["audio"].append(audio_paths[utt_id])
        manifest["n_frames"].append(audio_lengths[utt_id])
        manifest["tgt_text"].append(tgt_utt)
        manifest["speaker"].append(speaker_id)
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), root / f"{split}_{task}.tsv")


# Collect train text to generate sentencepiece model and vocabulary later on
train_text = pd.read_csv(root / "train_st_en_de.tsv", sep="\t")["tgt_text"].tolist()


with open(root / "train_text.txt", "w") as f:
    for t in train_text:
        f.write(t + "\n")


# Train sentencepiece model and generate subword vocabulary

# The vocab size depends on your dataset size . Since we are using a very small
# subset of the data for demonstration, we set a small vocab size of 900
VOCAB_SIZE = "800"
VOCAB_TYPE = "unigram"
spm_filename_prefix = f"spm_{VOCAB_TYPE}{VOCAB_SIZE}_{task}"

gen_vocab(
    Path(root / "train_text.txt"),
    root / spm_filename_prefix,
    VOCAB_TYPE,
    VOCAB_SIZE,
)


gen_config_yaml(
    root,
    spm_filename=spm_filename_prefix + ".model",
    yaml_filename=f"config_{task}.yaml",
)
