import argparse
import os
from pathlib import Path
from deep_translator import GoogleTranslator
import sentencepiece as spm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch

from examples.speech_to_text.data_utils import load_df_from_tsv


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


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


NUM_SAMPLES_PER_PREDICTION = 10

LLM_POST_EDITING_PROMPT = """[INST] <<SYS>>
You are a professional specialized in ASR (Automatic Speech Recognition) transcription enhancement.
<</SYS>>
Task:
1. Punctuation Restoration: Add necessary punctuation to make the sentences grammatically correct and more readable.
2. Evaluate the hypotheses for accuracy and likelihood of being the correct representation of the spoken text.
3. Select the best hypothesis that accurately reflects the original spoken text.

Input list of the 10 best ASR hypotheses:
1. "mr jones said meet me at 10 am in the conference room"
2. "mister jones said meet me at ten a m in the conference room"
3. "mr jones said meet me at ten am in the conference room"
4. "mister jones said meet me at 10 am in the conference room"
5. "mr jones said meet me at ten in the morning in the conference room"
6. "mr jones said meet me at 10 in the conference room"
7. "mister jones said meet me at ten in the conference room"
8. "mr jones said meet me at 10 in the conference room"
9. "mr jones said meet me at ten a m in the conference room"
10. "mister jones said meet me at 10 a m in the conference room"[/INST]
Best Hypothesis: "Mr. Jones said, 'Meet me at 10 a.m. in the conference room.'"

</s><s>[INST]
Input list of the 10 best ASR hypotheses:
{HYPOTHESES}[/INST]
Best Hypothesis:"""

log("Loading LLaMA...")

LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Using device: {DEVICE}")
log("Loading Tokenizer")

TOKENIZER = AutoTokenizer.from_pretrained(LLAMA_MODEL, padding_side="left")
TOKENIZER.pad_token = TOKENIZER.eos_token

log("Tokenizer ready.")
log("Loading LLaMA model.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

LLM = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, quantization_config=bnb_config, device_map="auto")

log("Configuring LLaMA model.")

LLM.config.pad_token_id = TOKENIZER.pad_token_id = 0  # unk
LLM.config.bos_token_id = 1  # bos
LLM.config.eos_token_id = 2  # eos

log("LLaMA ready.")


def cleanup_output(output: str) -> str:
    output = output.split("\n")[0].strip()
    output = output.replace("  ", " ")
    output = output.replace(" ,", ", ")

    # the output should always be in 'Best Hypothesis: "..."' format so we can just extract the text in between the quotes
    output = output.split('"')[1]

    return output


def generate(prompts: list[str]) -> list[str]:
    model_inputs = TOKENIZER(prompts, return_tensors="pt", padding=True).to(DEVICE)

    generation_config = GenerationConfig(
        temperature=0.2,  # TODO really low temperature but still get hallucinations
        top_p=0.75,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=1,
    )

    max_new_tokens = model_inputs.input_ids.shape[-1] * 2

    with torch.inference_mode():
        generated_ids = LLM.generate(
            **model_inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )

    decoded_outputs = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
    decoded_inputs = TOKENIZER.batch_decode(model_inputs.input_ids, skip_special_tokens=True)

    cleaned_outputs = [
        cleanup_output(output.replace(prompt, ""))
        for output, prompt in zip(decoded_outputs, decoded_inputs)
    ]

    with open("llama_outputs.txt", "a", encoding="utf-8") as f:
        for output, prompt in zip(cleaned_outputs, decoded_inputs):
            f.write(f"{prompt}\n{output}\n\n")
            f.write("-" * 100 + "\n\n")

    return cleaned_outputs


def sample_print(data_list: list[str]) -> None:
    DELIMITER = "----------------------------------------"
    log(DELIMITER)
    for data in data_list[:2]:
        log(data)
    log(DELIMITER)


def encode_and_save(lines: list[str], output_file: str) -> list[str]:
    spm_model = spm.SentencePieceProcessor(model_file=SPM_OUTPUT_FILE.as_posix())

    lines = [
        " ".join(spm_model.encode(line.strip(), out_type=str))
        for line in lines
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return lines


def process_reference_file():
    with open(args.ref_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[:20]
    lines = process_reference_text(lines)

    log("Processed reference file")
    sample_print(lines)

    encoded = encode_and_save(lines, args.ref_output_file)

    log("Encoded reference file")
    sample_print(encoded)

    log("Done processing reference file!")


def process_hypothesis_file():
    log("Processing hypothesis file...")
    with open(args.hyp_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[:20*NUM_SAMPLES_PER_PREDICTION]
    lines = process_hypothesis_text(lines)

    log("Processed hypothesis file")
    sample_print(lines)

    encoded = encode_and_save(lines, args.hyp_output_file)

    log("Encoded hypothesis file")
    sample_print(encoded)

    log("Done processing hypothesis file!")


def process_reference_text(lines):
    # TODO this should be temporary as we should get a ST dataset in the future
    log("Translating file...")

    # text_mapping = {"original": [], "target": []}
    text_mapping = load_df_from_tsv(ASR_ROOT / "text.tsv")

    text_mapping = {
        target: original.lower().strip()
        for original, target in zip(text_mapping["original"], text_mapping["target"])
    }

    # this should at least be the original reference text without any processing, which we do during ASR preprocessing
    lines = [
        text_mapping[line] if line in text_mapping else line.lower().strip()
        for line in lines
    ]

    translator = GoogleTranslator(source=args.src_lng, target=args.target_lng)

    log(f"Translating... (target language: {args.target_lng})")

    translated = translator.translate_batch(lines)

    log("Translated file")
    sample_print(translated)

    return translated


def process_hypothesis_text(lines):
    # process hypothesis text, for example, add punctuation, capitalization, etc.
    # use a LLM to post-process the hypothesis text

    processed_lines = []

    # process in batches of 10
    batch_size = 10
    batch = []
    for i in tqdm(range(0, len(lines), NUM_SAMPLES_PER_PREDICTION), desc="Processing batches"):
        samples = lines[i:i + NUM_SAMPLES_PER_PREDICTION]

        formatted_samples = ""
        for i, sample in enumerate(samples):
            formatted_samples += f"{i + 1}. \"{sample}\"\n"

        prompt = LLM_POST_EDITING_PROMPT.format(HYPOTHESES=formatted_samples)
        batch.append(prompt)

        if len(batch) == batch_size:
            processed_lines.extend(generate(batch))
            batch = []

    if len(batch) > 0:
        processed_lines.extend(generate(batch))

    return processed_lines

    # for i in range(0, len(lines), 10):
    #     log(f"Processing lines {i} to {i + 10}...")
    #     hypotheses = lines[i:i + 10]
    #     prompt = LLM_POSTEDITING_PROMPT.format(HYPOTHESES="\n".join(hypotheses))
    # 
    #     max_hypothesis_length = max([len(hypothesis) for hypothesis in hypotheses])
    #     max_length = (len(prompt) + max_hypothesis_length) * 1.5
    # 
    #     # LLaMA model inference
    #     post_processed = generate([prompt])[0]
    # 
    #     processed_lines.append(post_processed)
    # 
    # return processed_lines


if __name__ == "__main__":
    log("Starting processing...")

    process_hypothesis_file()

    process_reference_file()

    log("Done processing!")
