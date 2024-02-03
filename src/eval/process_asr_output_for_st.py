import argparse
import os
import subprocess
from typing import Callable
from tqdm import tqdm
from fuzzywuzzy import process

from src.datasets.util import iterate_over_dataset

from src.mt.bpe import BPE, write_lines
from src.asr.config import cleanup_utterance
from src.logger_utils import get_logger
from src.datasets.concrete.covost import CoVoST, CoVoSTWithText

from src.paths import COVOST_ROOT, MT_SPM_MODEL, PUNCTUATION_SPM_MODEL, MT_ROOT


logger = get_logger("Eval::process_asr_output_for_st")

parser = argparse.ArgumentParser()
parser.add_argument("--ref_input_file", type=str)
parser.add_argument("--ref_output_file", type=str)
parser.add_argument("--hyp_input_file", type=str)
parser.add_argument("--hyp_output_file", type=str)
parser.add_argument("--type_of_postprocessing", type=str, default="llama") # "none", "llama", "custom"
parser.add_argument("--num_samples_per_prediction", type=int, default=10)
parser.add_argument("--num_samples_to_evaluate", type=int, default=100)
parser.add_argument("--src_lng", type=str, default="en")
parser.add_argument("--target_lng", type=str, default="de")
args = parser.parse_args()


def no_postprocessing(lines: list[str]) -> list[str]:
    return __get_only_best_hypothesis(lines)

def llama_postprocessing(lines: list[str]) -> list[str]:
    from src.llama.llama import generate
    
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
Best Hypothesis: \""""
    
    processed_lines = []

    # process in batches of 10
    batch_size = 10
    batch = []
    for i in tqdm(range(0, len(lines), args.num_samples_per_prediction), desc="Processing batches"):
        samples = lines[i:i + args.num_samples_per_prediction]

        formatted_samples = "\n".join(f'{i + 1}. "{sample}"' for i, sample in enumerate(samples))

        prompt = LLM_POST_EDITING_PROMPT.format(HYPOTHESES=formatted_samples)
        batch.append(prompt)

        if len(batch) == batch_size:
            processed_lines.extend(generate(batch))
            batch = []

    if len(batch) > 0:
        processed_lines.extend(generate(batch))

    def cleanup_output(output: str) -> str:
        output = output.split("\n")[0].strip() # only get the first line of the output
        last = output
        while last != output:
            last = output
            output = output.replace("  ", " ") # remove double spaces
            output = output.replace(" ,", ", ") # remove spaces before commas

        # the output should always be in 'Best Hypothesis: "..."' format so we can just extract the text in between the quotes
        output = output.split('"')[0]

        return output

    return [cleanup_output(line) for line in processed_lines]

def custom_postprocessing(lines: list[str]) -> list[str]:
    # Function to match and reorder lines
    def match_and_reorder_lines(ref_lines: list[str], processed_lines: list[str]) -> list[str]:
        matched_lines = []
        for processed_line in processed_lines:
            # Use fuzzy matching to find the best match for each processed line in ref_lines
            best_match = process.extractOne(processed_line, ref_lines)
            if best_match is not None:
                matched_lines.append(best_match[0])  # Add the best matching line
        return matched_lines
    TRAIN_WORKSPACE=MT_ROOT / "train" / "train_punctuation_covost"
    BINARY_DATA_DIR=TRAIN_WORKSPACE / "binarized_dataset"
    MODEL_DIR=TRAIN_WORKSPACE / "models"
    PREDICTIONS_DIR = f"{os.environ['HOME']}/predictions/eval_st/punctuation"
    TEST_PREF = PREDICTIONS_DIR + "/test"
    BEAM_SIZE = 16
    DECODE_BPE = "NONE"
    MT_BINARY_DATA_DIR= MT_ROOT / "train" / "finetune_mt_covost" / "binarized_dataset"
    
    
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                    
    only_best_hypothesis = __get_only_best_hypothesis(lines)
    BPE.write_encoded_lines(PUNCTUATION_SPM_MODEL, only_best_hypothesis, TEST_PREF + ".en")
    
    ref_lines = PROCESSED_LINES[args.ref_output_file]
    bpe_encoded_ref_lines = BPE.write_encoded_lines(PUNCTUATION_SPM_MODEL, ref_lines, TEST_PREF + ".de")
    bpe_decoded_ref_lines = BPE.from_pretrained(PUNCTUATION_SPM_MODEL).decode_lines(bpe_encoded_ref_lines)
    
    # copy PUNCTUATION_SPM_MODEL to PREDICTIONS_DIR
    os.system(f"cp {PUNCTUATION_SPM_MODEL.as_posix().replace('.model', '')}.* {PREDICTIONS_DIR}/")
    
    # call generate on the file
    COMMAND = f"./src/bash/translate_mt.sh {BINARY_DATA_DIR}/dict.en.txt {BINARY_DATA_DIR}/dict.de.txt {TEST_PREF} {PREDICTIONS_DIR} {MODEL_DIR.as_posix()} {PREDICTIONS_DIR} {BEAM_SIZE} {DECODE_BPE}"
    subprocess.run([COMMAND], shell=True)
    
    # read the predictions
    with open(PREDICTIONS_DIR + "/hyp_mt.txt", "r", encoding="utf-8") as f:
        predictions = BPE.from_pretrained(PUNCTUATION_SPM_MODEL).decode_lines([line.strip() for line in f.readlines()])
            
    with open(PREDICTIONS_DIR + "/ref_mt.txt", "r", encoding="utf-8") as f:
        ref_lines_in_order_of_processed = [line.strip() for line in f.readlines()]
        decoded_ref_lines_in_order_of_processed = BPE.from_pretrained(PUNCTUATION_SPM_MODEL).decode_lines(ref_lines_in_order_of_processed)
        PROCESSED_LINES[args.ref_output_file] = match_and_reorder_lines(ref_lines, decoded_ref_lines_in_order_of_processed)
        print("--------------------------------------------------------------")
        print("ref_lines_in_order_of_processed", ref_lines_in_order_of_processed)
        print("--------------------------------------------------------------")
        print("bpe_encoded_ref_lines", bpe_encoded_ref_lines)
        print("--------------------------------------------------------------")
        print("bpe_decoded_ref_lines", bpe_decoded_ref_lines)
        print("--------------------------------------------------------------")
        print("ref_lines", ref_lines)
        print("--------------------------------------------------------------")
        print("PROCESSED_LINES[args.ref_output_file]", PROCESSED_LINES[args.ref_output_file])
        print("--------------------------------------------------------------")
        # for line in ref_lines_in_order_of_processed:
        #     # find the best matching line in the original reference lines
        #     PROCESSED_LINES[args.ref_output_file].append(ref_lines[bpe_decoded_ref_lines.index(line)])
            
    return predictions    


TYPES_OF_POSTPROCESSING = {
    "none": no_postprocessing,
    "llama": llama_postprocessing,
    "custom": custom_postprocessing,
}

PROCESSED_LINES: dict[str, list[str]] = {}

def __get_only_best_hypothesis(lines: list[str]) -> list[str]:
    return [lines[i] for i in range(0, len(lines), args.num_samples_per_prediction)]

def __encode_and_save(lines: list[str], output_file: str) -> list[str]:
    bpe = BPE.from_pretrained(MT_SPM_MODEL)
    
    encoded_lines = bpe.encode_lines(lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(encoded_lines))
    
    return encoded_lines

def __process_file(
    input_file: str, 
    output_file: str, 
    num_samples:int, 
    process_text_function: Callable[[list[str]], list[str]], 
    log_message: str
) -> None:
    """
    Generic function to process files.
    
    :param input_file: The path to the input file.
    :param output_file: The path to the output file.
    :param num_samples: The number of samples to process.
    :param process_text_function: The function to process the text.
    :param log_message: The message to log for processing.
    """
    def __sample_print(data_list: list[str]) -> None:
        logger.info("----------------------------------------")
        for data in data_list[:5]:
            logger.info(data)
        logger.info("----------------------------------------")

    logger.info(f"Processing {log_message} file...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines_subsample = lines[:num_samples]
    lines_processed = process_text_function(lines_subsample)

    logger.info(f"Processed {log_message} file")
    __sample_print(lines_processed)
    
    PROCESSED_LINES[output_file] = lines_processed

    logger.info(f"Done processing {log_message} file!")


def process_reference_text(lines: list[str]) -> list[str]:
    covost = CoVoSTWithText(COVOST_ROOT, CoVoST.SPLITS[2], "en", "de")
    
    translation_dict = {
        cleanup_utterance(original): target
        for original, target in iterate_over_dataset(covost, desc="Creating translation dictionary")
    }
    
    logger.info("Translating file...")

    translations = [
        translation_dict[line] if line in translation_dict else line.lower().strip()
        for line in lines
    ]

    return translations


def process_hypothesis_text(lines: list[str]) -> list[str]:
    """Post-processes the hypothesis text. For example, add punctuation, capitalization, etc. Returns a list of post-processed hypothesis texts in the same order as the input lines."""
    return TYPES_OF_POSTPROCESSING[args.type_of_postprocessing](lines)


if __name__ == "__main__":
    logger.info("Starting processing...")

    # process the reference file
    __process_file(
        args.ref_input_file, 
        args.ref_output_file, 
        args.num_samples_to_evaluate, 
        process_reference_text, 
        "reference"
    )

    # process the hypothesis file
    num_samples = args.num_samples_to_evaluate * args.num_samples_per_prediction
    __process_file(
        args.hyp_input_file, 
        args.hyp_output_file, 
        num_samples, 
        process_hypothesis_text, 
        "hypothesis"
    )

    logger.info("Done processing!")
    
    logger.info("Writing processed lines to file...")
    
    min_length = min(len(v) for v in PROCESSED_LINES.values())

    for output_file, lines_processed in PROCESSED_LINES.items():
        if len(lines_processed) > min_length:
            logger.warning("-------------------------------------------------")
            logger.warning(f"Output file {output_file} has more lines than the minimum length of {min_length}.")
            logger.warning("Truncating the file to the minimum length.")
            logger.warning("-------------------------------------------------") 
        
        __encode_and_save(lines_processed[:min_length], output_file)
