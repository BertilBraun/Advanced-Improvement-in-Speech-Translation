# Advanced Improvement in Speech Translation

## Overview

This project, developed as part of our university coursework, focuses on advancing the field of speech-to-text translation through comprehensive experimentation with automatic speech recognition (ASR) and machine translation (MT). Our goal is to improve the understanding and processing of spoken languages using neural network models.

## Motivation

With approximately 7000 spoken languages worldwide and the European Union advocating for citizens to know at least two languages besides their mother tongue, the need for robust speech translation tools has never been more apparent. Our project aims to contribute to this global need by enhancing automatic speech translation capabilities.

## Project Structure

```text
.
├── documentation
│   └── (Project reports, additional documentation)
├── src
│   ├── train
│   ├── eval
│   ├── logs
│   ├── datasets
│   └── ... (Other directories)
├── CLUSTER_README.md (Guide on BWUniCluster setup and usage)
├── environment.yml (Conda environment file)
├── README.md
└── requirements.txt
```

## Approach

Our approach is a cascaded system combining ASR and MT to translate spoken language into text. This method allows modular development, detailed error analysis, and flexibility in using different datasets and models.

### Evaluation Metrics

We utilized various metrics to evaluate our models, including Word Error Rate (WER), BLEU (BiLingual Evaluation Understudy Score), BertScore, and COMET (Crosslingual Optimized Metric for Evaluation of Translation), ensuring a comprehensive assessment of performance.

## Experiments and Results

Our experiments leveraged the CoVoST 2 dataset, focusing on the English to German translation task. We explored several architectures and techniques, including:

- Convolution-augmented transformer models for ASR.
- Simple transformer models for MT, enhanced with paraphrase generation and a Cosine Learning Rate Scheduler.
- Techniques to improve the connection from ASR to MT, including custom models and prompts for LLaMa2.

Results indicate significant insights into the cascaded approach's effectiveness and areas for improvement in speech translation systems.

## Conclusions

The project highlighted the challenges of speech translation, including the need for high-quality data and the complexity of integrating ASR and MT systems. Future work will focus on exploring larger datasets, more advanced models, and innovative data augmentation techniques to enhance performance further.

## Further Reading and Documentation

For a more detailed explanation of our methods, experiments, and results, please refer to the [`documentation`](/documentation) folder and the [`CLUSTER_README.md`](/CLUSTER_README.md) for setup and usage instructions on the BWUniCluster. Checkout the [presentation](/documentation/Presentation.pdf) for a overview and the results of our work.
