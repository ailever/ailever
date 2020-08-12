# built-in / external modules
import argparse

# torch
import torch

# ailever modules
from visualization import AileverVisualizer


def load():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='ailever')
    parser.add_argument('--device', type=str, default=device)
    
    
    # training information
    parser.add_argument('--epochs',
                        type=int,
                        default=5)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)

    # datasets path
    parser.add_argument('--dataset_name',
                        type=str,
                        default='MNIST')
    parser.add_argument('--dataset_loadpath',
                        type=str,
                        default='torch')
    parser.add_argument('--dataset_savepath',
                        type=str,
                        default='datasets/')
    parser.add_argument('--xlsx_path',
                        type=str,
                        default='datasets/dataset.xlsx')
    parser.add_argument('--json_path',
                        type=str,
                        default='datasets/dataset.json')
    parser.add_argument('--pkl_path',
                        type=str,
                        default='datasets/dataset.pkl')
    parser.add_argument('--hdf5_path',
                        type=str,
                        default='datasets/dataset.hdf5')
    
    # visualization
    parser.add_argument('--server',
                        type=str,
                        default='http://localhost')
    parser.add_argument('--port',
                        type=int,
                        default=8097)
    parser.add_argument('--env',
                        type=str,
                        default='main')


    options = parser.parse_args()
    setattr(options, 'vis', AileverVisualizer(options))
    return options

    
