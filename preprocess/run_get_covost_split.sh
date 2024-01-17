#!/bin/bash

echo "Starting the script execution."

source ../setup.sh
echo "Setup script sourced successfully."

ASR="/pfs/work7/workspace/scratch/ubppd-ASR"
COVOST="$ASR/covost"
VALIDATED="$COVOST/corpus-16.1/en/validated.tsv"

export COVOST_ROOT="$COVOST/data/translations"
echo "COVOST_ROOT environment variable set to $COVOST_ROOT."

# Define the file you want to check
FILE_TO_CHECK="$COVOST_ROOT/covost_v2.en_de.test.tsv"
echo "File to check for existence: $FILE_TO_CHECK."

# Check if the file exists
if [ ! -f "$FILE_TO_CHECK" ]; then
    echo "File $FILE_TO_CHECK not found. Running Python script to generate TSV files."
    
    # Run the Python script with the file
    python "$COVOST/get_covost_splits.py" \
      --version 2 --src-lang en --tgt-lang de \
      --root "$COVOST_ROOT" \
      --cv-tsv "$VALIDATED"

    echo "Python script to get the covost splits are completed. TSV files should be generated in the $COVOST_ROOT directory."
else
    echo "File $FILE_TO_CHECK already exists."
fi



echo "Running python script: process_audio_covost.py"
python process_audio_covost.py


echo "Script execution finished."
