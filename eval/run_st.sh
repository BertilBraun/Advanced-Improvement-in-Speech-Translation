#!/bin/bash

#SBATCH --job-name=PST_eval_st             # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=01:15:00                    # wall-clock time limit  
#SBATCH --mem=60000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=../../ST/logs/eval_output_sbatch_%j.txt
#SBATCH --error=../../ST/logs/eval_error_sbatch_%j.txt


# TODO We might want to train multiple models with different hyperparameters and then use an ensemble of them for evaluation.
# TODO Before evaluation, we might want to run a LLM on the generated output, to possibly improve the results.
# TODO Use of a beam search decoder might also improve the results.


# Take Away for Speech Data Augmentation:
# Data augmentation converts over-fit problem to under-fit problems. 
# From below figures, you can notice that the model without augmentation (None) perform nearly perfect in training set while no similar result is performed in other dataset.
# https://github.com/makcedward/nlpaug if that wants to be tried out.



source ../setup.sh

asr_model="wav2vec" # "wav2vec" or "mel"

ASR_DATA_DIR=~/ST/data/ASR
ASR_MODEL_DIR=~/ASR/models/$asr_model

PRED_OUTPUT_DIR=~/ST/predictions
ASR_PRED_LOG=$PRED_OUTPUT_DIR/en_s2t.pred.log

mkdir -p $PRED_OUTPUT_DIR

echo "Starting ASR prediction for $asr_model"

fairseq-generate $ASR_DATA_DIR \
    --config-yaml config.yaml --gen-subset test-clean \
    --task speech_to_text \
    --path $ASR_MODEL_DIR/checkpoint_best.pt \
    --max-tokens 50000 --beam 5 --scoring wer > $ASR_PRED_LOG

echo "Prediction done for $asr_model"

grep ^H $ASR_PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp_asr.txt
grep ^T $ASR_PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref_asr.txt

echo "Prediction files written for $asr_model"
echo "Sample predictions:"

head $PRED_OUTPUT_DIR/hyp_asr.txt

echo "Sample references:"

head $PRED_OUTPUT_DIR/ref_asr.txt

echo "WER ASR:"
tail -n 1 $PRED_LOG



MT_DATA_DIR=~/ST/data/MT
MT_MODEL_DIR=~/MT/models

MT_PRED_LOG=$PRED_OUTPUT_DIR/en-de.pred.log

echo "Starting translation..."

fairseq-preprocess \
    --source-lang en --target-lang de \
    --testpref $PRED_OUTPUT_DIR/hyp_asr.txt \
    --destdir $MT_DATA_DIR/eval.de-en \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8

fairseq-generate $MT_DATA_DIR/eval.de-en \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $MT_MODEL_DIR/checkpoint_best.pt \
      --batch-size 256 \
      --beam 4 \
      --remove-bpe=sentencepiece > $MT_PRED_LOG

echo "Translations done"

grep ^H $MT_PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp_mt.txt
grep ^T $MT_PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref_mt.txt

echo "Translations files written"
echo "Sample predictions:"

head $PRED_OUTPUT_DIR/hyp_mt.txt

echo "Sample references:"

head $PRED_OUTPUT_DIR/ref_mt.txt

echo "WER:"
tail -n 1 $MT_PRED_LOG

echo "BLEU:"
cat $PRED_OUTPUT_DIR/hyp_mt.txt | sacrebleu $PRED_OUTPUT_DIR/ref_mt.txt