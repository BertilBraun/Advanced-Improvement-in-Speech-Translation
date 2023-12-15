# Project Setup and Execution Guide

1. Connect to the cluster via SSH

    ```bash
    ssh <username>@uc2.scc.kit.edu
    ```

    The very first action should be, to create a shared workspace:

    ```bash
    ws_allocate ASR 60 # 60 Days
    ws_allocate MT 60 # 60 Days
    ```

    Add users to the workspace:

    ```bash
    setfacl -Rm u:USERNAME:rwX,d:u:USERNAME:rwX $(ws_find ASR)
    setfacl -Rm u:USERNAME:rwX,d:u:USERNAME:rwX $(ws_find MT)
    ```

    To access the workspace, run:

    ```bash
    cd $(ws_find ASR)
    cd $(ws_find MT)
    ```

    To check the remaining time of the workspace, run:

    ```bash
    ws_list
    ```

    To extend the workspace, run:

    ```bash
    ws_extend ASR 30 # 30 Days
    ws_extend MT 30 # 30 Days
    ```

    Note that this is automatically done in the `setup.sh` script once the workspace is about to expire.

2. Download the project

    ```bash
    git clone https://github.com/BertilBraun/Advanced-Improvement-in-Speech-Translation.git PST
    ```

3. Create a virtual environment

    First, install miniconda by following the instructions [here](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install).

    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

    # Initialize conda in your bash shell
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    ```

    Then, create a virtual environment and install the required packages:

    ```bash
    cd ~/PST
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

    Once you are sure that the script is executable and runs without errors, you can submit it to the cluster.

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
    #SBATCH --output=../../ASR/logs/output_%j.txt   # standard output and error log
    #SBATCH --error=../../ASR/logs/error_%j.txt     # %j is the job id, making each log file unique, therefore not overwriting each other
    ```

    To then submit the script to the cluster, run:

    ```bash
    sbatch run_YOUR_SCRIPT.sh
    ```

6. Monitoring

    To monitor the status of your job, run:

    ```bash
    squeue -u <username>
    ```

    To cancel a job, run:

    ```bash
    scancel <job-id>
    ```

    The logs of the job are stored in the `~/ASR/logs` directory or `~/MT/logs` directory, depending on the task. The output and error logs are named `output_<job-id>.txt` and `error_<job-id>.txt`, respectively. The `job-id` is the number that is returned when submitting the job to the cluster.

7. Downloading the results

    To download the results from the cluster, run:

    ```bash
    scp [-r] <username>@uc2.scc.kit.edu:~/<PATH-ON-REMOTE> <LOCAL-PATH>
    ```

    The `-r` flag is only required if you want to download a directory.
