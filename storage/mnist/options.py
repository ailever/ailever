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
                        default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)

    # load_paths
    parser.add_argument('--dataset_path',
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
    
    # additional arguments
    options = parser.parse_args()
    setattr(options, 'vis', AileverVisualizer(options.epochs, ['train', 'validation']))
    return options

    
