from typing import Literal
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import re
import os
import sys

DATA_TYPE = dict[Literal['epoch', 'loss', 'lr', 'valid_loss'], list[float | int]]


def find_all_files(paths: list[str]) -> list[str]:
    all_file_paths = []
    for path in paths:
        if os.path.isdir(path):
            assert len(paths) == 1, 'Only one directory path is allowed'
            # Add all files in the directory to the list
            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path) and (filename.endswith('.log') or filename.endswith('.txt')):
                    all_file_paths.append(full_path)
        elif os.path.isfile(path):
            all_file_paths.append(path)
    return all_file_paths


def extract_data_from_logs(file_paths: list[str]) -> DATA_TYPE:
    data: dict[int, tuple[float, float, float]] = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                if 'INFO | train |' in line and 'loss' in line:
                    epoch = int(re.search(r'epoch (\d+)', line).group(1))  # type: ignore
                    loss = float(re.search(r'loss (\d+\.\d+)', line).group(1))  # type: ignore
                    lr = float(re.search(r'lr (\d+\.\d+(e[-+]\d+)?)', line).group(1))  # type: ignore

                    _, __, valid_loss = data.get(epoch, (0, 0, 0))
                    data[epoch] = (loss, lr, valid_loss)
                elif '| valid on ' in line and 'loss' in line:
                    epoch = int(re.search(r'epoch (\d+)', line).group(1))  # type: ignore
                    valid_loss = float(re.search(r'loss (\d+\.\d+)', line).group(1))  # type: ignore

                    loss, lr, _ = data.get(epoch, (0, 0, 0))
                    data[epoch] = (loss, lr, valid_loss)

    # Convert the data to a dictionary of lists
    data_dict: DATA_TYPE = {'epoch': [], 'loss': [], 'lr': [], 'valid_loss': []}
    for epoch, (loss, lr, valid_loss) in data.items():
        data_dict['epoch'].append(epoch)
        data_dict['loss'].append(loss)
        data_dict['lr'].append(lr)
        if valid_loss != 0:
            data_dict['valid_loss'].append(valid_loss)
        else:
            average = ((data[epoch - 1][2] if epoch > 1 else 0) + (data[epoch + 1][2] if epoch < len(data) else 0)) / 2
            data_dict['valid_loss'].append(average)

    return data_dict


def plot_data(data: DATA_TYPE) -> Figure:
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
    return fig


if __name__ == '__main__':
    file_paths = sys.argv[1:]
    if file_paths == []:
        print('Please provide file paths')
        sys.exit(1)

    if file_paths[0] == 'help':
        print('Usage: python plot_loss.py <path_to_log_file1> <path_to_log_file2> ...')
        print('Or: python plot_loss.py <path_to_log_folder>')
        sys.exit(0)

    all_log_file_paths = find_all_files(file_paths)
    data = extract_data_from_logs(all_log_file_paths)
    # print(data)
    fig = plot_data(data)

    # if file_paths[0] is a directory, then save the plot in the directory
    if os.path.isdir(file_paths[0]):
        fig.savefig(os.path.join(file_paths[0], 'loss_plot.png'))
