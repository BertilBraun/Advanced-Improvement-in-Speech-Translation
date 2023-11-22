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

export OMP_NUM_THREADS=40

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

source ~/env/bin/activate  # Activate your virtual environment.

git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
cd ..

export PYTHONPATH="~/fairseq/:$PYTHONPATH"

# TODO Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/encoded_dataset/
DATA_DIR=~/fairseq/examples/speech_to_text/data

# TODO Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/models/
MODEL_DIR=~/fairseq/examples/speech_to_text/models

# create folders if they don't exist
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

# if DATA_DIR is empty, print error and return
if [ -z "$(ls -A $DATA_DIR)" ]; then
  echo "Error: DATA_DIR is empty. Run preprocess/run_process_audio.sh first."
  exit 1
fi

# TODO adjust paths and num-workers

# Train the model in parallel
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR --save-dir $MODEL_DIR \
  --train-subset train-clean-100 --valid-subset dev-clean \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8