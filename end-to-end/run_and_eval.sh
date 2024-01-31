#!/bin/bash

#SBATCH --job-name=process_end_to_end      # job name
#SBATCH --partition=gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=16:00:00                    # wall-clock time limit  
#SBATCH --mem=200000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=process_end_to_end_logs_%j.txt
#SBATCH --error=process_end_to_end_logs_%j.txt

source ../setup.sh

#pip install pandas torchaudio sentencepiece  #should already be done in environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#Data Preparation
DATA_ROOT="content/fairseq/examples/speech_to_text/data"
mkdir -p $DATA_ROOT

#run data preparation
python data_prep.py #>> data_prep_$SLURM_JOB_ID.txt 2>&1

#Training
WORKSPACE_ROOT=/pfs/work7/workspace/scratch/uxude-ASR
TRAIN_ROOT=$WORKSPACE_ROOT/train/st_end_to_end

COVOST_ROOT=$WORKSPACE_ROOT/dataset/covost/en
ST_SAVE_DIR=$TRAIN_ROOT/model
PRED_OUTPUT_DIR=$TRAIN_ROOT/prediction
PRED_LOG=$PRED_OUTPUT_DIR/st_en_de.pred.log


mkdir -p $ST_SAVE_DIR
mkdir -p $PRED_OUTPUT_DIR


fairseq-train $COVOST_ROOT \
  --config-yaml config_st_en_de.yaml --train-subset train_st_en_de --valid-subset dev_st_en_de \
  --save-dir $ST_SAVE_DIR \
  --save-interval-updates 50000\
  --num-workers 4 --max-update 30000 --max-tokens 50000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --encoder-freezing-updates 1000 --optimizer adam --lr 2e-3 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8



# Inference & Evaluation
fairseq-generate $COVOST_ROOT \
  --config-yaml config_st_en_de.yaml --gen-subset test_st_en_de --task speech_to_text \
  --path $ST_SAVE_DIR/checkpoint_best.pt \
  --max-tokens 50000 --beam 5 --scoring sacrebleu > $PRED_LOG

grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

head $PRED_OUTPUT_DIR/hyp.txt
head $PRED_OUTPUT_DIR/ref.txt

tail -n 1 $PRED_LOG

