import pyter
import argparse


def calculate_ter(hyp_file: str, ref_file: str) -> float:
    with open(hyp_file, 'r') as hf, open(ref_file, 'r') as rf:
        hyp_lines = hf.readlines()
        ref_lines = rf.readlines()

    if len(hyp_lines) != len(ref_lines):
        raise ValueError("The number of lines in hypothesis and reference files must be the same")

    total_ter = 0.0
    for hyp_line, ref_line in zip(hyp_lines, ref_lines):
        hyp_tokens = hyp_line.strip().split()
        ref_tokens = ref_line.strip().split()
        total_ter += pyter.ter(hyp_tokens, ref_tokens)

    average_ter = total_ter / len(hyp_lines)
    return average_ter


parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, required=True)
parser.add_argument("--ref_file", type=str, required=True)
args = parser.parse_args()

print("Average TER Score:", calculate_ter(args.hyp_file, args.ref_file))
