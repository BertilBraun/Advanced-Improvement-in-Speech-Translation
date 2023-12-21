import argparse
from deep_translator import GoogleTranslator
import sentencepiece as spm


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
        
    spm_output_model = spm.SentencePieceProcessor(model_file=args.spm_output_file)
    
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
        
    spm_input_model = spm.SentencePieceProcessor(model_file=args.spm_input_file)
    spm_output_model = spm.SentencePieceProcessor(model_file=args.spm_output_file)
    
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
    
    translator = GoogleTranslator(source=args.src_lng, target=args.target_lng) 

    print(f"Translating... (target language: {args.target_lng})")

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