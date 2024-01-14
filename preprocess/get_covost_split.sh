#!/bin/bash

source ../setup.sh

COVOST="/pfs/work7/workspace/scratch/ubppd-ASR/covost"
TRANSLATIONS="/pfs/work7/workspace/scratch/ubppd-ASR/covost/data/tt"
VALIDATED="/pfs/work7/workspace/scratch/ubppd-ASR/covost/corpus-16.1/en"

cd "$COVOST" || exit

# Run the Python script with direct path constructions
python get_covost_splits.py \
  --version 2 --src-lang en --tgt-lang de \
  --root "$TRANSLATIONS" \
  --cv-tsv "$VALIDATED"
