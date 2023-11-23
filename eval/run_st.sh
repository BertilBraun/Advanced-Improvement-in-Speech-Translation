# Run the input speach through the asr model
# Run the output text through the mt model


# TODO We might want to train multiple models with different hyperparameters and then use an ensemble of them for evaluation.
# TODO Before evaluation, we might want to run a LLM on the generated output, to possibly improve the results.
# TODO Use of a beam search decoder might also improve the results.


# Take Away for Speech Data Augmentation:
# Data augmentation converts over-fit problem to under-fit problems. 
# From below figures, you can notice that the model without augmentation (None) perform nearly perfect in training set while no similar result is performed in other dataset.
# https://github.com/makcedward/nlpaug if that wants to be tried out.