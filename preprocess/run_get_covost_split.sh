#!/bin/bash

source ../setup.sh

ASR="/pfs/work7/workspace/scratch/ubppd-ASR/"
COVOST="$ASR/covost"
ROOT="$COVOST/data/translations"
VALIDATED="$COVOST/corpus-16.1/en/validated.tsv"

cd "$COVOST" || exit

# Run the Python script with direct path constructions
python get_covost_splits.py \
  --version 2 --src-lang en --tgt-lang de \
  --root "$ROOT" \
  --cv-tsv "$VALIDATED"


# This will result in
# You should get 3 TSV files (covost.<src_lang_code>_<tgt_lang_code>.<split>.tsv) for train, dev and test splits,
# respectively in --root Directory.
#   Each of them has 4 columns: 
#   path (audio filename), 
#   sentence (transcript), 
#   translation and client_id (speaker ID).
