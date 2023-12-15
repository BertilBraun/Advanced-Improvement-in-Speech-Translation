# Project Setup and Execution Guide

1. Connect to the cluster via SSH

    ```bash
    ssh <username>@uc2.scc.kit.edu
    ```

2. Download the project

    ```bash
    git clone https://github.com/BertilBraun/Advanced-Improvement-in-Speech-Translation.git AI-ST
    ```

3. Create a virtual environment

    First, install miniconda by following the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install).

    Then, create a virtual environment and install the required packages:

    ```bash
    cd ~/AI-ST
    conda create --name nmt
    conda activate nmt
    # Install required packages from environment.yml
    conda env update -f environment.yml
    # Ensure setup completed and install additional packages
    ./setup.sh
    ```

4. Running

    Ensure that the scripts start executing on the login node to avoid errors after the job has been submitted to the cluster.

    ```bash
    ./run_YOUR_SCRIPT.sh
    ```

    Ensure that the script is executable:

    ```bash
    chmod +x run_YOUR_SCRIPT.sh
    ```

5. Submitting to the cluster

    Once you are sure that the script is executable and runs without errors, you can submit it to the cluster:

    ```bash
    sbatch run_YOUR_SCRIPT.sh
    ```

    Make sure, to have set the correct `SBATCH` parameters in the script, such as `timeouts`, required cluster `cores` and `GPUs` and the `job-name`. Ensure, that the correct and required modules are being loaded by calling the `~/AI-ST/setup.sh` script.

    ```bash
    #SBATCH --job-name=PST_process_audio            # job name
    #SBATCH --partition=gpu_4                       # single, gpu_4
    #SBATCH --time=02:00:00                         # wall-clock time limit  
    #SBATCH --mem=200000                            # in MB check limits per node
    #SBATCH --nodes=1                               # number of nodes to be used
    #SBATCH --cpus-per-task=1                       # number of CPUs required per MPI task
    #SBATCH --ntasks-per-node=1                     # maximum count of tasks per node
    #SBATCH --mail-type=ALL                         # Notify user by email when certain event types occur.
    #SBATCH --gres=gpu:4                            # number of GPUs required per node 
    #SBATCH --output=~/PST/ASR/logs/output_%j.txt   # standard output and error log
    #SBATCH --error=~/PST/ASR/logs/error_%j.txt     # %j is the job id, making each log file unique, therefore not overwriting each other
    ```
