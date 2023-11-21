from pathlib import Path
from typing import Literal
from mpi4py import MPI
import itertools
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/text_translation/dataset/
DATASET_FOLDER = Path('sample_data')
DATASET_FOLDER.mkdir(exist_ok=True)
    
# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/text_translation/paraphrased_dataset/
OUTPUT_FOLDER = Path('sample_data')
OUTPUT_FOLDER.mkdir(exist_ok=True)

DATASET_URL = 'https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5'
    
DATASET_EN = DATASET_FOLDER / 'train.de-en.en'
DATASET_DE = DATASET_FOLDER /'train.de-en.de'    

OUTPUT_EN_FILE = OUTPUT_FOLDER / f'output_{rank}.en'
OUTPUT_DE_FILE = OUTPUT_FOLDER / f'output_{rank}.de'

def ensure_dataset_loaded() -> None:
    if rank == 0 and (not DATASET_EN.is_file() or not DATASET_DE.is_file()):
           
        download_location = DATASET_FOLDER / 'data.zip'
        # Download and unzip command
        download_command = f"wget -nc -O '{download_location}' '{DATASET_URL}'"
        unzip_command = f"unzip -o '{download_location}' -d '{DATASET_FOLDER}'"

        # Execute download and unzip commands
        print("Checking and downloading dataset...")
        os.system(download_command)
        print("Unzipping dataset...")
        os.system(unzip_command)
        print("Dataset ready.")
        
    comm.Barrier()
    
  

def generate_paraphrases(sentence: str, language: Literal('en') | Literal('de')) -> list[str]:
    # Placeholder for actual paraphrase generation with LLaMA 2
    # Replace with real paraphrase generation code
    # TODO
    return ["paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4"]

def read_dataset_segment(file_path: str, start_line: int, end_line: int) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for i, line in enumerate(file) if start_line <= i < end_line]

def main() -> None:
    ensure_dataset_loaded()
    
    num_lines = sum(1 for _ in open(DATASET_EN, 'r', encoding='utf-8'))
    lines_per_process = num_lines // size
    start_line = rank * lines_per_process
    end_line = num_lines if rank == size - 1 else start_line + lines_per_process

    english_sentences = read_dataset_segment(DATASET_EN, start_line, end_line)
    german_sentences = read_dataset_segment(DATASET_DE, start_line, end_line)

    # Open files once at the beginning
    with open(OUTPUT_EN_FILE, 'w', encoding='utf-8') as en_file, open(OUTPUT_DE_FILE, 'w', encoding='utf-8') as de_file:
        for en, de in zip(english_sentences, german_sentences):
            en_paraphrases = [en] + generate_paraphrases(en, 'en')
            de_paraphrases = [de] + generate_paraphrases(de, 'de')

            # Generate all combinations of English and German paraphrases and write directly to files
            for en_p, de_p in itertools.product(en_paraphrases, de_paraphrases):
                en_file.write(f"{en_p}\n")
                de_file.write(f"{de_p}\n")

if __name__ == "__main__":
    main()
