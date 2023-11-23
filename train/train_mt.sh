#!/bin/bash

# TODO Can this be run on the GPU?

#SBATCH --job-name=PST_train_asr           # job name
#SBATCH --partition=multiple               # TODO mby GPU queue for the resource allocation.
#SBATCH --time=30:00                       # TODO wall-clock time limit  
#SBATCH --mem=40000                        # TODO memory per node
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uxude@student.kit.edu  # notification email address

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VENVDIR=~/env                       # Export path to your virtual environment.
export PYDIR=.     			                   # Export path to directory containing Python script.

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

source ${VENVDIR}/bin/activate  # Activate your virtual environment.

git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
cd ..

export PYTHONPATH="~/fairseq/:$PYTHONPATH"

# TODO Something like: /pfs/work7/workspace/scratch/uxxxx-PST/text_translation/paraphrased_dataset/
DATA_DIR=~/fairseq/examples/translation/sample_data

# TODO Something like: /pfs/work7/workspace/scratch/uxxxx-PST/text_translation/binarized_dataset/
BINARY_DATA_DIR=~/fairseq/examples/translation/sample_data

# TODO Something like: /pfs/work7/workspace/scratch/uxxxx-PST/text_translation/models/
MODEL_DIR=~/fairseq/examples/translation/models

# create folders if they don't exist
mkdir -p $DATA_DIR
mkdir -p $BINARY_DATA_DIR
mkdir -p $MODEL_DIR

# if DATA_DIR is empty, print error and return
if [ -z "$(ls -A $DATA_DIR)" ]; then
  echo "Error: DATA_DIR is empty. Run preprocess/run_generate_paraphrases.sh first."
  exit 1
fi

# Binarize the data for training
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref "$DATA_DIR/spm.train.de-en" \
    --validpref "$DATA_DIR/spm.dev.de-en" \
    --testpref "$DATA_DIR/spm.tst.de-en" \
    --destdir "$BINARY_DATA_DIR/iwslt14.de-en" \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8

# TODO adjust paths and num-workers

# Train the model in parallel
fairseq-train \
    "$BINARY_DATA_DIR/iwslt14.de-en" --save-dir $MODEL_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 2 \
    --max-tokens 4096 \
    --max-epoch 20 \
    --fp16
