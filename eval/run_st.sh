#!/bin/bash

#SBATCH --job-name=eval_st                 # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=01:15:00                    # wall-clock time limit  
#SBATCH --mem=60000                        # memory per node
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

ASR_DATA_DIR=~/ASR/wav2vec
ASR_MODEL_DIR=~/ASR/models/wav2vec

PRED_OUTPUT_DIR=~/ST/predictions
ASR_PRED_LOG=$PRED_OUTPUT_DIR/en_s2t.pred.log


MT_DATA_DIR=~/ST/predictions/MT/eval.de-en
MT_MODEL_DIR=~/MT/models

MT_PRED_LOG=$PRED_OUTPUT_DIR/en-de.pred.log

mkdir -p $PRED_OUTPUT_DIR
mkdir -p $ASR_DATA_DIR

echo "Starting ASR prediction for wav2vec"

fairseq-generate $ASR_DATA_DIR \
    --config-yaml config.yaml --gen-subset test-clean \
    --task speech_to_text \
    --path $ASR_MODEL_DIR/checkpoint_best.pt \
    --max-tokens 50000 --beam 10 --scoring wer \
    --nbest 10 > $ASR_PRED_LOG

echo "Prediction done for wav2vec"

grep ^H $ASR_PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp_asr.txt
grep ^T $ASR_PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref_asr.txt

# TODO how is the output structured?

echo "Prediction files written for wav2vec"
echo "Sample predictions:"

head -2 $PRED_OUTPUT_DIR/hyp_asr.txt

echo "Sample references:"

head -2 $PRED_OUTPUT_DIR/ref_asr.txt

echo "WER ASR:"
tail -n 1 $ASR_PRED_LOG


python process_asr_output_for_st.py \
    --ref_input_file $PRED_OUTPUT_DIR/ref_asr.txt \
    --ref_output_file $PRED_OUTPUT_DIR/asr_out.de \
    --hyp_input_file $PRED_OUTPUT_DIR/hyp_asr.txt \
    --hyp_output_file $PRED_OUTPUT_DIR/asr_out.en


echo "Starting translation..."

# TODO srcdict and tgtdict need to be set to the correct paths
fairseq-preprocess \
    --source-lang en --target-lang de \
    --srcdict ~/MT/initial_binarized_dataset/iwslt14.de-en/dict.en.txt \
    --tgtdict ~/MT/initial_binarized_dataset/iwslt14.de-en/dict.de.txt \
    --testpref $PRED_OUTPUT_DIR/asr_out \
    --destdir $MT_DATA_DIR \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8

fairseq-generate $MT_DATA_DIR \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $MT_MODEL_DIR/checkpoint_best.pt \
      --batch-size 64 \
      --beam 16 \
      --remove-bpe=sentencepiece > $MT_PRED_LOG

echo "Translations done"

grep ^H $MT_PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp_mt.txt
grep ^T $MT_PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref_mt.txt

echo "Translations files written"
echo "Sample predictions:"

head -2 $PRED_OUTPUT_DIR/hyp_mt.txt

echo "Sample references:"

head -2 $PRED_OUTPUT_DIR/ref_mt.txt

echo "WER:"
tail -n 1 $MT_PRED_LOG

echo "BLEU:"
cat $PRED_OUTPUT_DIR/hyp_mt.txt | sacrebleu $PRED_OUTPUT_DIR/ref_mt.txt