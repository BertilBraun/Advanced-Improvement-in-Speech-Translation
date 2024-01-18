#!/bin/bash

#SBATCH --job-name=process_end_to_end      # job name
#SBATCH --partition=gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=17:00:00                    # wall-clock time limit  
#SBATCH --mem=200000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=process_end_to_end_logs_%j.txt
#SBATCH --error=process_end_to_end_logs_%j.txt

source ../setup.sh

pip list -v                                #could be removed

#import torch python ....

#git clone https://github.com/facebookresearch/fairseq.git  #should already be done in environment setup

#cd fairseq
#pip install --editable ./ #should already be done in environment setup

#echo $PYTHONPATH
#import os python  #should already be done in environment setup
#os.environ['PYTHONPATH'] += ":/content/fairseq/"  #should already be done in environment setup
#echo $PYTHONPATH

#pip install pandas torchaudio sentencepiece  #should already be done in environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#Data Preparation
# Define folder path to store the data
DATA_ROOT="content/fairseq/examples/speech_to_text/data"
# Create folder if not yet exist
#mkdir -p $DATA_ROOT
# Download the subset data
#wget -O $DATA_ROOT/covost2-subset.tar.gz https://bwsyncandshare.kit.edu/s/ePneqHCHKCgFT9N/download/covost2-subset.tar.gz
# Unzip it
#tar -xf $DATA_ROOT/covost2-subset.tar.gz -C $DATA_ROOT

#run data preparation
#python data_prep.py #>> data_prep_$SLURM_JOB_ID.txt 2>&1

#Training
export COVOST_ROOT=/pfs/work7/workspace/scratch/ubppd-ASR/covost/corpus-16.1
export ST_SAVE_DIR=content/fairseq/examples/speech_to_text/model


#mkdir -p $ST_SAVE_DIR
#CUDA_VISIBLE_DEVICES=0 fairseq-train $COVOST_ROOT/en \
#  --config-yaml config_st_en_de.yaml --train-subset train_st_en_de --valid-subset dev_st_en_de \
#  --save-dir $ST_SAVE_DIR \
#  --save-interval-updates 50000\
#  --num-workers 4 --max-update 30000 --max-tokens 50000 \
#  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
#  --arch s2t_transformer_s --encoder-freezing-updates 1000 --optimizer adam --lr 2e-3 \
#  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8



# Inference & Evaluation
export PRED_OUTPUT_DIR=content/fairseq/examples/speech_to_text/pred_eval
export PRED_LOG=content/fairseq/examples/speech_to_text/pred_eval/st_en_de.pred.log
mkdir -p $PRED_OUTPUT_DIR
fairseq-generate $COVOST_ROOT/en \
  --config-yaml config_st_en_de.yaml --gen-subset test_st_en_de --task speech_to_text \
  --path $ST_SAVE_DIR/checkpoint_best.pt \
  --skip-invalid-size-inputs-valid-test \
  --max-tokens 50000 --beam 5 --scoring sacrebleu > $PRED_LOG


grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

head $PRED_OUTPUT_DIR/hyp.txt
head $PRED_OUTPUT_DIR/ref.txt

tail -n 1 $PRED_LOG

#Saving local files to google drive not necessary for now