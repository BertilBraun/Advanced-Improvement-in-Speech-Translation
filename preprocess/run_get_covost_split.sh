#!/bin/bash

source ../setup.sh

ASR="/pfs/work7/workspace/scratch/ubppd-ASR/"
COVOST="$ASR/covost"
TRANSLATIONS="$COVOST/data/tt"
VALIDATED="$COVOST/corpus-16.1/en/validated.tsv"

cd "$COVOST" || exit

# Run the Python script with direct path constructions
python get_covost_splits.py \
  --version 2 --src-lang en --tgt-lang de \
  --root "$TRANSLATIONS" \
  --cv-tsv "$VALIDATED"
