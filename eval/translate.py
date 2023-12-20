import argparse
from deep_translator import GoogleTranslator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--src_lng", type=str, default="en")
    parser.add_argument("--target_lng", type=str, default="de")
    args = parser.parse_args()
    
    # Use the Google Translate API to translate the input file
    # and write the output to the output file.
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    
    translator = GoogleTranslator(source=args.src_lng, target=args.target_lng) 

    print(f"Translating... (target language: {args.target_lng})")

    translated = [
        translator.translate(line)
        for line in lines
    ]

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(translated))