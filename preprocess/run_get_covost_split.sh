#!/bin/bash

echo "Starting the script execution."

source ../setup.sh
echo "Setup script sourced successfully."

export ASR="/pfs/work7/workspace/scratch/ubppd-ASR"
export COVOST="$ASR/covost"
export COVOST_CORPUS="$COVOST/corpus-16.1"

export COVOST_DATA="$COVOST/data"
echo "COVOST_DATA environment variable set to $COVOST_DATA."

# Define the file you want to check
FILE_TO_CHECK="$COVOST_DATA/translations/covost_v2.en_de.test.tsv"
echo "File to check for existence: $FILE_TO_CHECK."

# Check if the file exists
if [ ! -f "$FILE_TO_CHECK" ]; then
    echo "File $FILE_TO_CHECK not found. Running Python script to generate TSV files."
    
    # Run the Python script with the file
    python "$COVOST/get_covost_splits.py" \
      --version 2 --src-lang en --tgt-lang de \
      --root "$COVOST_DATA/translations" \
      --cv-tsv "$COVOST_CORPUS/en/validated.tsv"

    echo "Python script to get the covost splits are completed. TSV files should be generated in the $COVOST_DATA directory."
else
    echo "File $FILE_TO_CHECK already exists."
fi

cd "$HOME/Advanced-Improvement-in-Speech-Translation" || { echo "Could not change directory. Exiting script ..."; exit 1;}

echo "Running python script: process_audio_covost.py"
python -m preprocess.process_audio_covost.py


echo "Script execution finished."
