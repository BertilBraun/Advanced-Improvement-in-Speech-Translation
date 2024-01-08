
import os
import sentencepiece as spm

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

ROOT_FOLDER = Path(f"{os.environ['HOME']}") / "MT"

WORKSPACE_ROOT_FOLDER = Path("/pfs/work7/workspace/scratch/uxude-MT")

PARAPHRASED_EN_FILE = ROOT_FOLDER / "output" / "train_paraphrased.de-en.en"
PARAPHRASED_DE_FILE = ROOT_FOLDER / "output" / "train_paraphrased.de-en.de"

OUTPUT_EN_FILE = WORKSPACE_ROOT_FOLDER / "output" / "train_complete.de-en.en"
OUTPUT_DE_FILE = WORKSPACE_ROOT_FOLDER / "output" / "train_complete.de-en.de"

SPM_OUTPUT_EN_FILE = WORKSPACE_ROOT_FOLDER / "output" / "spm.train_complete.de-en.en"
SPM_OUTPUT_DE_FILE = WORKSPACE_ROOT_FOLDER / "output" / "spm.train_complete.de-en.de"


WRITE_DATASET = True

ALLOWED_NON_ASCII_CHARS = "–“’‘„”�…€—βüöäÜÖÄ"

def data_generator():
    # WMT 14-19 datasets, but generate a set from these to avoid duplicates
    s = set()    
    total_sizes = 100_000_000 # approximate size of the dataset (100 million)
    
    for i in range(14, 20):
        log(f"Loading WMT{i} dataset...")
        dataset = load_dataset(f"wmt{i}", "de-en", split="train", trust_remote_code=True)
        log(f"Now processing WMT{i} dataset...")
        # log(f"Dataset length: {len(dataset['translation'])}")
        # total_sizes += len(dataset["translation"])
        
        # extend the set with the new sentences
        s.update(
            (sentence["en"], sentence["de"]) for sentence in dataset["translation"]
        )

    # convert the set to a list
    s = list(s)
    log(f"The total size of the dataset is {total_sizes}.")
    log(f"Filtered dataset length: {len(s)}")
    log(f"Removed {total_sizes - len(s)} duplicates.")
    
    return s


def translation_pair_check(en, de):
    # only keep sentences that are only ascii characters
    def ascii_check(s):
        return all(ord(c) < 256 or c in ALLOWED_NON_ASCII_CHARS for c in s)
    
    return ascii_check(en) and ascii_check(de)


def cleanup(en, de):
    def clean(s):
        return s.replace("\n", " ").replace("\t", " ").strip()
    
    return clean(en), clean(de)    
    

if __name__ == "__main__":
    
    if WRITE_DATASET:
        skipped_lines = 0
        log("Merging paraphrased and original datasets...")
    
        with open(OUTPUT_DE_FILE, "w", encoding="utf-8") as f_de, \
            open(OUTPUT_EN_FILE, "w", encoding="utf-8") as f_en, \
            open(PARAPHRASED_EN_FILE, "r", encoding="utf-8") as f_paraphrased_en, \
            open(PARAPHRASED_DE_FILE, "r", encoding="utf-8") as f_paraphrased_de:
            
            for en, de in tqdm(zip(f_paraphrased_en.readlines(), f_paraphrased_de.readlines()), desc="Merging datasets"):
                en, de = cleanup(en, de)
                if translation_pair_check(en, de):
                    f_en.write(en + "\n")
                    f_de.write(de + "\n")
                else:
                    skipped_lines += 1
                
            for en, de in tqdm(data_generator(), desc="Merging WMT datasets"):
                en, de = cleanup(en, de)
                if translation_pair_check(en, de):
                    f_en.write(en + "\n")
                    f_de.write(de + "\n")
                else:
                    skipped_lines += 1
                
        log(f"Skipped {skipped_lines} lines because they contained non-ascii characters.")

    spm.SentencePieceTrainer.train(input=f"{OUTPUT_DE_FILE},{OUTPUT_EN_FILE}",
                                model_prefix="bpe",
                                vocab_size=5000,
                                input_sentence_size=1000000,
                                shuffle_input_sentence=True)

    log('Finished training sentencepiece model.')
    
    spm_model = spm.SentencePieceProcessor(model_file="bpe.model")
    
    log("BPE model ready.")
    
    for data_file, spm_file in zip((OUTPUT_DE_FILE, OUTPUT_EN_FILE), (SPM_OUTPUT_DE_FILE, SPM_OUTPUT_EN_FILE)):
        
        with open(data_file, "r", encoding="utf-8") as f_in, \
            open(spm_file, "w", encoding="utf-8") as f_out:
            file_name = data_file.split("/")[-1]
            for line in tqdm(f_in.readlines(), desc=f"Segmenting '{file_name}' Dataset"):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                f_out.write(" ".join(line_segmented) + "\n")
