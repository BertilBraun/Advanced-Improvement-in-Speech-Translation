import argparse
import os
from pathlib import Path
from deep_translator import GoogleTranslator
import sentencepiece as spm

from examples.speech_to_text.data_utils import load_df_from_tsv

parser = argparse.ArgumentParser()
parser.add_argument("--ref_input_file", type=str)
parser.add_argument("--ref_output_file", type=str)
parser.add_argument("--hyp_input_file", type=str)
parser.add_argument("--hyp_output_file", type=str)
parser.add_argument("--spm_input_file", type=str)
parser.add_argument("--spm_output_file", type=str)
parser.add_argument("--src_lng", type=str, default="en")
parser.add_argument("--target_lng", type=str, default="de")
args = parser.parse_args()
     
     
HOME = Path(f"{os.environ['HOME']}")
ASR_ROOT = HOME / "ASR/wav2vec"
SPM_INPUT_FILE = ASR_ROOT / "spm_unigram900.model"
SPM_OUTPUT_FILE = HOME / "PST/train/bpe.model"

def sample_print(data_list):
    DELIMITER = "----------------------------------------"
    print(DELIMITER)
    for data in data_list[:2]:
        print(data)
    print(DELIMITER)


def process_reference_file():
    with open(args.ref_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        
    lines = process_reference_text(lines)
        
    spm_output_model = spm.SentencePieceProcessor(model_file=SPM_OUTPUT_FILE)
    
    lines = [
        " ".join(spm_output_model.encode(line.strip(), out_type=str))
        for line in lines
    ]
    
    print("Encoded output file")
    sample_print(lines)

    with open(args.ref_output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_hypothesis_file():
    print("Processing hypothesis file...")
    with open(args.hyp_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        
    spm_input_model = spm.SentencePieceProcessor(model_file=SPM_INPUT_FILE)
    spm_output_model = spm.SentencePieceProcessor(model_file=SPM_OUTPUT_FILE)
    
    lines = [
        spm_input_model.decode(line.split())
        for line in lines
    ]
    
    print("Decoded input file")
    sample_print(lines)
    
    lines = process_hypothesis_text(lines)
        
    lines = [
        " ".join(spm_output_model.encode(line.strip(), out_type=str))
        for line in lines
    ]
    
    print("Encoded output file")
    sample_print(lines)
            
    with open(args.hyp_output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            
    print("Done processing hypothesis file!")


def process_reference_text(lines):
    # TODO this should be temporary as we should get a ST dataset in the future
    # TODO otherwise, this should at least be the original reference text without any processing, which we do during ASR preprocessing
    print("Translating file...")


    # text_mapping = {"original": [], "target": []}
    text_mapping = load_df_from_tsv(ASR_ROOT / "text.tsv")
    
    text_mapping = {
        target: original
        for original, target in zip(text_mapping["original"], text_mapping["target"])
    }
    
    translator = GoogleTranslator(source=args.src_lng, target=args.target_lng) 

    print(f"Translating... (target language: {args.target_lng})")

    lines = [
        text_mapping[line] if line in text_mapping else line
        for line in lines
    ]
    translated = translator.translate_batch(lines)
    
    print("Translated file")
    sample_print(translated)
    
    return translated


def process_hypothesis_text(lines):
    # TODO process hypothesis text, for example, add punctuation, capitalization, etc.
    # TODO use a LLM to post-process the hypothesis text
    return lines
        


if __name__ == "__main__":
    print("Starting processing...")
    
    process_reference_file()
    
    process_hypothesis_file()
    
    print("Done processing!")