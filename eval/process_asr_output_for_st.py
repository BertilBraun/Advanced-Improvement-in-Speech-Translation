import argparse
from deep_translator import GoogleTranslator
import sentencepiece as spm



def translate_file(input_file, output_file, src_lng, target_lng, spm_output_model_file):
    print("Translating file...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        
    lines = lines[:50]
    
    translator = GoogleTranslator(source=src_lng, target=target_lng) 

    print(f"Translating... (target language: {target_lng})")

    translated = [
        translator.translate(line)
        for line in lines
    ]
    
    print("Translated file")
    for line in translated[:2]:
        print(line)
        
        
    spm_output_model = spm.SentencePieceProcessor(model_file=spm_output_model_file)
    
    translated = [
        " ".join(spm_output_model.encode(line.strip(), out_type=str))
        for line in translated
    ]
    
    print("Encoded output file")
    for line in translated[:2]:
        print(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(translated))
        
    print("Done translating file!")


def process_hypothesis_file(input_file, output_file, spm_input_model_file, spm_output_model_file):
    print("Processing hypothesis file...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        
    lines = lines[:50]
        
    spm_input_model = spm.SentencePieceProcessor(model_file=spm_input_model_file)
    spm_output_model = spm.SentencePieceProcessor(model_file=spm_output_model_file)
    
    lines = [
        spm_input_model.decode(line.split())
        for line in lines
    ]
    
    print("Decoded input file")
    for line in lines[:2]:
        print(line)
        
    lines = [
        " ".join(spm_output_model.encode(line.strip(), out_type=str))
        for line in lines
    ]
    
    print("Encoded output file")
    for line in lines[:2]:
        print(line)
            
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            
    print("Done processing hypothesis file!")

if __name__ == "__main__":

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
    
    # Use the Google Translate API to translate the input file
    # and write the output to the output file.
    
    print("Starting processing...")
    
    translate_file(args.ref_input_file, args.ref_output_file, args.src_lng, args.target_lng, args.spm_output_file)
    
    process_hypothesis_file(args.hyp_input_file, args.hyp_output_file, args.spm_input_file, args.spm_output_file)
    
    print("Done processing!")