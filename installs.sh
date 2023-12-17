#!/bin/bash

#SBATCH --job-name=install                 # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=0:30:00                     # wall-clock time limit  
#SBATCH --mem=8000                         # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=log.txt
#SBATCH --error=log.txt

module purge
# module load devel/cuda/11.8

# conda initialize
cd ~/miniconda3/bin
./deactivate
./conda env remove -n pst
./conda create -y -n pst python=3.8
./activate pst

cd ~

python -m pip install --upgrade pip
pip3 install torch torchvision torchaudio Cython

python PST/cuda_test.py

rm -rf fairseq
echo "Fairseq: Cloning and installing..."

git clone https://github.com/pytorch/fairseq.git
cd fairseq
git submodule update --init --recursive

# in fairseq/models/speech_to_text/s2t_transformer.py replace "args.input_feat_per_channel * args.input_channels," with "768, # args.input_feat_per_channel * args.input_channels"
sed -i 's/args.input_feat_per_channel \* args.input_channels,/768, # args.input_feat_per_channel \* args.input_channels/' fairseq/models/speech_to_text/s2t_transformer.py

pip install --editable ./
python setup.py build_ext --inplace


export PYTHONPATH=~/fairseq/:$PYTHONPATH
export PATH=~/fairseq/:$PATH

cd ~/PST
pip install -r requirements.txt

echo "Setup complete. Starting script execution..."

cd train
./train_asr.sh