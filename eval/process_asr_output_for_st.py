import argparse
import os
from pathlib import Path
from deep_translator import GoogleTranslator
import requests
import sentencepiece as spm

from examples.speech_to_text.data_utils import load_df_from_tsv
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("--ref_input_file", type=str)
parser.add_argument("--ref_output_file", type=str)
parser.add_argument("--hyp_input_file", type=str)
parser.add_argument("--hyp_output_file", type=str)
parser.add_argument("--src_lng", type=str, default="en")
parser.add_argument("--target_lng", type=str, default="de")
args = parser.parse_args()
     
     
HOME = Path(f"{os.environ['HOME']}")
ASR_ROOT = HOME / "ASR/wav2vec"
SPM_INPUT_FILE = ASR_ROOT / "spm_unigram1000.model"
SPM_OUTPUT_FILE = HOME / "PST/train/bpe.model"

MODEL_PATH = HOME / "ST/llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q5_K_M.gguf"
)


LLM_POSTEDITING_PROMPT = """Enhance the following ASR output by adding punctuation and selecting the most probable transcription from the given hypotheses.

Input list of the 10 best ASR hypotheses:
{HYPOTHESES}

Task:
1. Punctuation Restoration: Go through each hypothesis and add necessary punctuation (periods, commas, question marks, etc.) to make the sentences grammatically correct and more readable. Consider natural pauses, intonation, and grammatical structure.
2. Hypothesis Evaluation: Evaluate each punctuated hypothesis for its likelihood to be the most accurate representation of the spoken text. Consider factors like context, grammar, and completeness of the sentences.
3. Selection of the Best Hypothesis: From the punctuated and evaluated hypotheses, select the one that is most likely to be the correct transcription of the original speech. 

Output the most likely and accurately punctuated ASR transcription:
"""


def ensure_model_loaded() -> None:
    if not MODEL_PATH.is_file():
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH.as_posix(), "wb") as file:
            file.write(response.content)
        print("Model downloaded.")


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
    print("Translating file...")

    # text_mapping = {"original": [], "target": []}
    text_mapping = load_df_from_tsv(ASR_ROOT / "text.tsv")
    
    text_mapping = {
        target: original
        for original, target in zip(text_mapping["original"], text_mapping["target"])
    }
    
    # this should at least be the original reference text without any processing, which we do during ASR preprocessing
    lines = [
        text_mapping[line] if line in text_mapping else line
        for line in lines
    ]
    
    translator = GoogleTranslator(source=args.src_lng, target=args.target_lng) 

    print(f"Translating... (target language: {args.target_lng})")

    translated = translator.translate_batch(lines)
    
    print("Translated file")
    sample_print(translated)
    
    return translated


def process_hypothesis_text(lines):
    # process hypothesis text, for example, add punctuation, capitalization, etc.
    # use a LLM to post-process the hypothesis text
    
    # LLaMA model initialization
    LLM = Llama(model_path=MODEL_PATH.as_posix(), n_ctx=2048)
    
    processed_lines = []
    
    for i in range(0, len(lines), 10):
        print(f"Processing lines {i} to {i+10}...")
        hypotheses = lines[i:i+10]
        prompt = LLM_POSTEDITING_PROMPT.format(HYPOTHESES="\n".join(hypotheses))
        
        max_hypothesis_length = max([len(hypothesis) for hypothesis in hypotheses])
        max_length = (len(prompt) + max_hypothesis_length) * 1.5
        
        # LLaMA model inference
        post_processed = LLM(prompt, max_length=0)["choices"][0]["text"]
        
        processed_lines.append(post_processed)
        
    return processed_lines
        


if __name__ == "__main__":
    print("Starting processing...")
    
    process_reference_file()
    
    process_hypothesis_file()
    
    print("Done processing!")