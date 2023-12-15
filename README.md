# Project Setup and Execution Guide

1. Connect to the cluster via SSH
```bash
ssh <username>@uc2.scc.kit.edu
```

2. Load the required modules
```bash
module purge
module load compiler/gnu/10.2
```

3. Download the project
```bash
git clone https://github.com/BertilBraun/Advanced-Improvement-in-Speech-Translation.git AI-ST
```

4. Create a virtual environment

First, install miniconda by following the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install).

Then, create a virtual environment and install the required packages from `requirements.txt`:
```bash
conda create --name nmt
conda activate nmt
# Install required packages from previous conda environment which was exported to environment.yml
conda env update -f environment.yml
```
