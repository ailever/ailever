# built-in / external modules
import argparse

# torch
import torch


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

    # load_paths
    parser.add_argument('--xlsx_path',
                        type=str,
                        default='dataset/dataset.xlsx')
    parser.add_argument('--json_path',
                        type=str,
                        default='dataset/dataset.json')
    parser.add_argument('--pkl_path',
                        type=str,
                        default='dataset/dataset.pkl')
    parser.add_argument('--hdf5_path',
                        type=str,
                        default='dataset/dataset.hdf5')
    
    options = parser.parse_args()
    return options
