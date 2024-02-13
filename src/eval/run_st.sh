#!/bin/bash

#SBATCH --job-name=eval_st                 # job name
#SBATCH --partition=dev_gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=eval_st_log_%j.txt
#SBATCH --error=eval_st_log_%j.txt

cd ../../
source setup.sh
PREDICTION_DIR=~/predictions/eval_st

POSTPROCESSING_TYPES=("custom" "llama" "none") # postprocessing types
NUM_SAMPLES_TO_EVALUATE=1000000000 # number of samples to evaluate (sth like 10 billion to evaluate all)

ASR_TEST_SUBSET=test
ASR_BEAM_SIZE=10
ASR_N_BEST=10

MT_BEAM_SIZE=16


python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi


echo "Transcribing the test set..."

# source src/bash/transcribe_asr.sh \
#         $ASR_TEST_SUBSET \
#         $ASR_DATA_DIR \
#         $ASR_MODEL_DIR \
#         $PREDICTION_DIR \
#         $ASR_BEAM_SIZE \
#         $ASR_N_BEST

echo "Transcription done"

for TYPE_OF_POSTPROCESSING in "${POSTPROCESSING_TYPES[@]}"; do
    echo "--------------------------------------------------"
    echo "Processing with postprocessing type: $TYPE_OF_POSTPROCESSING"
    echo "--------------------------------------------------"
    echo "Starting processing of ASR output for MT input..."

    POSTPROCESSING_PREDICTION_DIR=$PREDICTION_DIR/$TYPE_OF_POSTPROCESSING

    mkdir -p $POSTPROCESSING_PREDICTION_DIR

    # python -m src.eval.process_asr_output_for_st \
    #     --ref_input_file $PREDICTION_DIR/ref_asr.txt \
    #     --ref_output_file $POSTPROCESSING_PREDICTION_DIR/asr_out.de \
    #     --hyp_input_file $PREDICTION_DIR/hyp_asr.txt \
    #     --hyp_output_file $POSTPROCESSING_PREDICTION_DIR/asr_out.en \
    #     --type_of_postprocessing $TYPE_OF_POSTPROCESSING \
    #     --num_samples_per_prediction $ASR_N_BEST \
    #     --num_samples_to_evaluate $NUM_SAMPLES_TO_EVALUATE

    echo "Processing done"
    echo "--------------------------------------------------"
    echo "Starting translation..."

    # source src/bash/translate_mt.sh \
    #         $MT_PARAPHRASED_BINARY_DATA_DIR/dict.en.txt \
    #         $MT_PARAPHRASED_BINARY_DATA_DIR/dict.de.txt \
    #         $POSTPROCESSING_PREDICTION_DIR/asr_out \
    #         $POSTPROCESSING_PREDICTION_DIR \
    #         $MT_PARAPHRASED_MODEL_DIR \
    #         $POSTPROCESSING_PREDICTION_DIR \
    #         $MT_BEAM_SIZE

    echo "Processing completed for postprocessing type: $TYPE_OF_POSTPROCESSING"
done

echo "--------------------------------------------------"
echo "Results for the different postprocessing types"
echo "--------------------------------------------------"

# output the results of the different postprocessing types
for TYPE_OF_POSTPROCESSING in "${POSTPROCESSING_TYPES[@]}"; do
    POSTPROCESSING_PREDICTION_DIR=$PREDICTION_DIR/$TYPE_OF_POSTPROCESSING

    SRC=$POSTPROCESSING_PREDICTION_DIR/src_mt.txt
    REF=$POSTPROCESSING_PREDICTION_DIR/ref_mt.txt
    HYP=$POSTPROCESSING_PREDICTION_DIR/hyp_mt.txt

    
    source src/bash/extract_from_prediction.sh  \
        $POSTPROCESSING_PREDICTION_DIR/translate_mt_pred.log \
        $HYP \
        $REF \
        1 \
        $SRC

    echo "Postprocessing type: $TYPE_OF_POSTPROCESSING"
    echo "--------------------------------------------------"
    echo "BLEU score:"
    sacrebleu -tok none -s 'none' $REF < $HYP
    echo "--------------------------------------------------"
    echo "TER score:"
    python -m src.eval.calculate_ter --hyp_file $HYP --ref_file $REF
    echo "--------------------------------------------------"
    echo "BERTScore:"
    bert-score -r $REF -c $HYP --lang de
    echo "--------------------------------------------------"
    echo "COMET:"
    comet-score -s $SRC -r $REF -t $HYP --gpus 1
    echo "--------------------------------------------------"
done
