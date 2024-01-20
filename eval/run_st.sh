#!/bin/bash

#SBATCH --job-name=eval_st                 # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=03:00:00                    # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=eval_st_log_%j.txt
#SBATCH --error=eval_st_log_%j.txt


# TODO We might want to train multiple models with different hyperparameters and then use an ensemble of them for evaluation.
# TODO Before evaluation, we might want to run a LLM on the generated output, to possibly improve the results.
# TODO Use of a beam search decoder might also improve the results.


# Take Away for Speech Data Augmentation:
# Data augmentation converts over-fit problem to under-fit problems. 
# From below figures, you can notice that the model without augmentation (None) perform nearly perfect in training set while no similar result is performed in other dataset.
# https://github.com/makcedward/nlpaug if that wants to be tried out.



source ../setup.sh

WORKSPACE_ASR=/pfs/work7/workspace/scratch/uxude-ASR
WORKSPACE_MT=/pfs/work7/workspace/scratch/uxude-MT

ASR_DATA_DIR=~/ASR/wav2vec
ASR_MODEL_DIR=~/ASR/models/wav2vec

PRED_OUTPUT_DIR=~/ST/predictions
ASR_PRED_LOG=$PRED_OUTPUT_DIR/en_s2t.pred.log


MT_DATA_DIR=~/ST/predictions/MT/eval.de-en
MT_MODEL_DIR=~/MT/models

MT_BINARIZED_DATA_DIR=$WORKSPACE_MT/binarized_dataset

MT_PRED_LOG=$PRED_OUTPUT_DIR/en-de.pred.log

mkdir -p $PRED_OUTPUT_DIR
mkdir -p $ASR_DATA_DIR

echo "Starting ASR prediction for wav2vec"

# fairseq-generate $ASR_DATA_DIR \
#     --config-yaml config.yaml --gen-subset test-clean \
#     --task speech_to_text \
#     --path $ASR_MODEL_DIR/checkpoint_best.pt \
#     --max-tokens 50000 --beam 10 --scoring wer \
#     --nbest 10 > $ASR_PRED_LOG

echo "Prediction done for wav2vec"

source ../src/bash/extract_from_prediction.sh \
    $ASR_PRED_LOG \
    $PRED_OUTPUT_DIR/hyp_asr.txt \
    $PRED_OUTPUT_DIR/ref_asr.txt

python process_asr_output_for_st.py \
    --ref_input_file $PRED_OUTPUT_DIR/ref_asr.txt \
    --ref_output_file $PRED_OUTPUT_DIR/asr_out.de \
    --hyp_input_file $PRED_OUTPUT_DIR/hyp_asr.txt \
    --hyp_output_file $PRED_OUTPUT_DIR/asr_out.en


echo "Starting translation..."

fairseq-preprocess \
    --source-lang en --target-lang de \
    --srcdict $MT_BINARIZED_DATA_DIR/dict.en.txt \
    --tgtdict $MT_BINARIZED_DATA_DIR/dict.de.txt \
    --testpref $PRED_OUTPUT_DIR/asr_out \
    --destdir $MT_DATA_DIR \
    --thresholdtgt 0 --thresholdsrc 0

fairseq-generate $MT_DATA_DIR \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $MT_MODEL_DIR/checkpoint_best.pt \
      --batch-size 64 \
      --beam 16 \
      --remove-bpe=sentencepiece > $MT_PRED_LOG

echo "Translations done"

source ../src/bash/extract_from_prediction.sh \
    $MT_PRED_LOG \
    $PRED_OUTPUT_DIR/hyp_mt.txt \
    $PRED_OUTPUT_DIR/ref_mt.txt
