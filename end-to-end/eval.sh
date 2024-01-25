#!/bin/bash

#SBATCH --job-name=process_end_to_end      # job name
#SBATCH --partition=dev_gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=200000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=process_end_to_end_logs_%j.txt
#SBATCH --error=process_end_to_end_logs_%j.txt

source ../setup.sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#Data Preparation
# Define folder path to store the data
DATA_ROOT="content/fairseq/examples/speech_to_text/data"

WORKSPACE_ROOT=/pfs/work7/workspace/scratch/uxude-ASR
TRAIN_ROOT=$WORKSPACE_ROOT/train/st_end_to_end

COVOST_ROOT=$WORKSPACE_ROOT/dataset/covost/en
ST_SAVE_DIR=$TRAIN_ROOT/model
PRED_OUTPUT_DIR=$TRAIN_ROOT/prediction
PRED_LOG=$PRED_OUTPUT_DIR/st_en_de.pred.log


mkdir -p $ST_SAVE_DIR
mkdir -p $PRED_OUTPUT_DIR


fairseq-generate $COVOST_ROOT \
  --config-yaml config_st_en_de.yaml --gen-subset test_st_en_de --task speech_to_text \
  --path $ST_SAVE_DIR/checkpoint_best.pt \
  --max-tokens 50000 --beam 5 --scoring sacrebleu > $PRED_LOG


grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

head $PRED_OUTPUT_DIR/hyp.txt
head $PRED_OUTPUT_DIR/ref.txt

tail -n 1 $PRED_LOG
