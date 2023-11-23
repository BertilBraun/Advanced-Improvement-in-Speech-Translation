from pathlib import Path
from typing import Literal

import requests
from mpi4py import MPI
import itertools
import os
from llama_cpp import Llama
import spacy

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

DATASET_EN = DATASET_FOLDER / 'train.de-en.en'
DATASET_DE = DATASET_FOLDER /'train.de-en.de'    

OUTPUT_EN_FILE = OUTPUT_FOLDER / f'output_{rank}.en'
OUTPUT_DE_FILE = OUTPUT_FOLDER / f'output_{rank}.de'

DATASET_URL = 'https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5'
    
MODEL_PATH = "./llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q5_K_M.gguf"

# LANGUAGE = Literal('en') | Literal('de')

PROMPT = { # TODO Modify and play with this prompt to properly generate 5 good paraphrases
    'en': "Generate five distinct paraphrases of the following English sentence:\n'{}'\nParaphrases:",
    'de': "Erzeugen Sie fünf unterschiedliche Paraphrasen des folgenden deutschen Satzes:\n'{}'\nParaphrasen:"
}

NLP = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm')
}

def ensure_model_loaded() -> None:
    if rank == 0 and not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as file:
            file.write(response.content)
        print("Model downloaded.")
    
    comm.Barrier()


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
     

# Generates 5 paraphrases for the input sentence with LLaMA 2
def generate_paraphrases(LLM, sentence: str, language) -> list[str]:
    
    # Format the prompt with the given sentence
    formatted_prompt = PROMPT[language].format(sentence)

    # Generate response using LLaMA 2
    # max_tokens=0 removes the response size limit
    output = LLM(formatted_prompt, max_tokens=0)
    
    # Extract paraphrases from the response
    paraphrases_text = output["choices"][0]["text"]
    
    # Use Spacy's sentence boundary detection to extract paraphrases
    doc = NLP[language](paraphrases_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Heuristics to identify and extract paraphrases
    paraphrases = [sent for sent in sentences if heuristic_is_paraphrase(sent, sentence)]

    if len(paraphrases) < 5:
        print(f"Warning: Only {len(paraphrases)} paraphrases generated for '{sentence}'")

    # Ensure only five paraphrases are returned
    return paraphrases[:5]


def heuristic_is_paraphrase(candidate: str, original: str, language) -> bool:
    # Convert sentences to Spacy Doc objects
    doc_candidate = NLP[language](candidate)
    doc_original = NLP[language](original)

    # Semantic similarity (using Spacy's built-in similarity method)
    similarity_threshold = 0.7  # TODO Adjust this threshold as needed
    if doc_candidate.similarity(doc_original) < similarity_threshold:
        return False

    # Structural or lexical difference
    if candidate.lower() == original.lower() or doc_candidate.text_with_ws == doc_original.text_with_ws:
        return False

    # Length similarity (within a certain percentage difference)
    length_diff_threshold = 0.2  # TODO 20% length difference threshold
    if abs(len(candidate) - len(original)) / max(len(candidate), len(original)) > length_diff_threshold:
        return False

    return True


def read_dataset_segment(file_path: str, start_line: int, end_line: int) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for i, line in enumerate(file) if start_line <= i < end_line]

def main() -> None:
    ensure_dataset_loaded()
    ensure_model_loaded()
    
    # LLaMA model initialization
    LLM = Llama(model_path=MODEL_PATH, n_ctx=2048)
    
    # Find out the number of lines in the dataset and split them evenly between processes
    num_lines = sum(1 for _ in open(DATASET_EN, 'r', encoding='utf-8'))
    lines_per_process = num_lines // size
    start_line = rank * lines_per_process
    end_line = num_lines if rank == size - 1 else start_line + lines_per_process

    # Read the dataset segment for this process
    english_sentences = read_dataset_segment(DATASET_EN, start_line, end_line)
    german_sentences = read_dataset_segment(DATASET_DE, start_line, end_line)

    with open(OUTPUT_EN_FILE, 'w', encoding='utf-8') as en_file, open(OUTPUT_DE_FILE, 'w', encoding='utf-8') as de_file:
        # For each sentence pair in the dataset segment, generate paraphrases and write all combinations to files
        for en, de in zip(english_sentences, german_sentences):
            en_paraphrases = [en] + generate_paraphrases(LLM, en, 'en')
            de_paraphrases = [de] + generate_paraphrases(LLM, de, 'de')

            # Generate all combinations of English and German paraphrases and write directly to files
            for en_p, de_p in itertools.product(en_paraphrases, de_paraphrases):
                en_file.write(f"{en_p}\n")
                de_file.write(f"{de_p}\n")

if __name__ == "__main__":
    main()
