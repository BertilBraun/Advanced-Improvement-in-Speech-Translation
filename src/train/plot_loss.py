import matplotlib.pyplot as plt
import re
import os
import sys

def find_all_files(paths: list[str]) -> list[str]:
    all_file_paths = []
    for path in paths:
        if os.path.isdir(path):
            # Add all files in the directory to the list
            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path):
                    all_file_paths.append(full_path)
        elif os.path.isfile(path):
            all_file_paths.append(path)
    return all_file_paths

def extract_data_from_logs(file_paths: list[str]) -> dict[str, list[float | int]]:
    data = {'epoch': [], 'loss': [], 'valid_loss': [], 'lr': []}
    valid_loss_written_for_epoch = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                if 'INFO | train |' in line and 'loss' in line:
                    epoch = int(re.search(r'epoch (\d+)', line).group(1))
                    loss = float(re.search(r'loss (\d+\.\d+)', line).group(1))
                    lr = float(re.search(r'lr (\d+\.\d+(e[-+]\d+)?)', line).group(1))
                    
                    if epoch in data['epoch']:
                        continue
                    
                    data['epoch'].append(epoch)
                    data['loss'].append(loss)
                    data['lr'].append(lr)
                elif '| valid on ' in line and 'loss' in line:
                    epoch = int(re.search(r'epoch (\d+)', line).group(1))
                    valid_loss = float(re.search(r'loss (\d+\.\d+)', line).group(1))
                    
                    if epoch in valid_loss_written_for_epoch:
                        continue
                    
                    valid_loss_written_for_epoch[epoch] = True
                    data['valid_loss'].append(valid_loss)

    print(len(data['epoch']), len(data['loss']), len(data['valid_loss']), len(data['lr']))
    return data

def plot_data(data) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(data['epoch'], data['loss'], label='Training Loss', color='tab:red')
    ax1.plot(data['epoch'], data['valid_loss'], label='Validation Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    ax1.set_title('Training and Validation Loss')

    # Plotting Learning Rate
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate', color='tab:green')
    ax2.plot(data['epoch'], data['lr'], label='Learning Rate', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')
    ax2.set_title('Learning Rate')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_paths = sys.argv[1:]
    if file_paths == []:
        print("Please provide file paths")
        sys.exit(1)
        
    if file_paths[0] == 'help':
        print("Usage: python plot_loss.py <path_to_log_file1> <path_to_log_folder> <path_to_log_file2> ...")
        sys.exit(0)
    
    file_paths = find_all_files(file_paths)
    data = extract_data_from_logs(file_paths)
    print(data)
    plot_data(data)
