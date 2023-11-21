# Project Setup and Execution Guide

This guide outlines the steps for setting up and running the speech processing project in a Python virtual environment. The project uses MPI for parallel processing and leverages PyTorch and the Hugging Face `transformers` library for audio data processing.

## Prerequisites

- Python (version 3.7 or higher is recommended)
- MPI implementation (like OpenMPI or MPICH) installed on your system

## Setting Up the Virtual Environment

1. **Create a Virtual Environment**:
   Navigate to your project directory and run:
   ```bash
   python3 -m venv env
   ```

2. **Activate the Virtual Environment**:
   - On Windows, run:
     ```bash
     .\env\Scripts\activate
     ```
   - On Unix or MacOS, run:
     ```bash
     source env/bin/activate
     ```

3. **Install Dependencies**:
   Install the dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Script

1. **Execute the Script with MPI**:
   Use the `mpiexec` or `mpirun` command to run your script across multiple processes. For example:
   ```bash
   mpiexec -n 4 python your_script.py
   ```
   Replace `your_script.py` with the name of your script. The `-n 4` argument specifies using 4 processes; adjust this number based on your available compute resources.

2. **Execute the Script on the Cluster**:
    Make a copy of the `run_template.sh` script and modify it to run your script by replacing `<path to your python script>` with the path to your script and set the relevant `timeouts`, required cluster `cores` and `GPUs` and the `job-name`. Ensure, that the correct and required modules are being loaded. Then, submit the script to the cluster using the `sbatch` command.
    ```bash
    sbatch run_my_script.sh
    ```

3. **Deactivating the Environment**:
   Once you are done, you can deactivate the virtual environment by running:
   ```bash
   deactivate
   ```
