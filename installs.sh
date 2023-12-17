#!/bin/bash

#SBATCH --job-name=install                 # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=0:30:00                     # wall-clock time limit  
#SBATCH --mem=8000                         # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=NONE                   # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=log.txt
#SBATCH --error=log.txt

module purge
# module load devel/cuda/11.8

# conda initialize
cd ~/miniconda3/bin
conda init bash
conda deactivate
conda deactivate
conda deactivate
conda env remove -n pst
conda create -y -n pst python=3.7
conda activate pst

python --version
conda install -c conda-forge gcc_linux-64 gxx_linux-64 cmake ninja

python -m pip install --upgrade pip

cd ~/PST
pip install -r requirements.txt

python cuda_test.py

cd ~
rm -rf fairseq
echo "Fairseq: Cloning and installing..."

git clone https://github.com/pytorch/fairseq.git
cd fairseq
# git submodule update --init --recursive

# in fairseq/models/speech_to_text/s2t_transformer.py replace "args.input_feat_per_channel * args.input_channels," with "768, # args.input_feat_per_channel * args.input_channels"
sed -i 's/args.input_feat_per_channel \* args.input_channels,/768, # args.input_feat_per_channel \* args.input_channels/' fairseq/models/speech_to_text/s2t_transformer.py

echo "Starting fairseq build..."
echo "========================================================================"

python setup.py build_ext --inplace

echo "========================================================================"

#git submodule update --init --recursive
#pip install --editable .
pip install .

# copy fairseq install to site-packages
cp -r ~/fairseq/fairseq/ ~/miniconda3/envs/pst/lib/python3.8/site-packages/fairseq/
# copy fairseq example to site-packages
cp -r ~/fairseq/examples/ ~/miniconda3/envs/pst/lib/python3.8/site-packages/examples/

export PYTHONPATH=~/fairseq/:$PYTHONPATH
export PATH=~/fairseq/:$PATH

which python
python --version

cd ~/PST
pip install -r requirements.txt

echo "Setup complete. Starting script execution..."

cd train
./train_asr.sh