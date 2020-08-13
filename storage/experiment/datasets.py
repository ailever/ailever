# built-in / external modules
import json
import pickle
import h5py
import numpy as np
import pandas as pd
import sklearn.datasets as skl_datasets
from sklearn.model_selection import train_test_split
from nlp import list_datasets
from nlp import load_dataset

# torch
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms

# ailever modules
import options
obj = type('obj', (), {})


class NLPDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.dataset = load_dataset(self.options.dataset_name, cache_dir=self.options.dataset_savepath)
        self.train_dataset = self.dataset['train'];     print(self.train_dataset._info.features)
        self.test_dataset = self.dataset['validation']; print(self.test_dataset._info.features)
        self.validation_dataset = obj()

        self.train_dataset.x = None
        self.train_dataset.y = None
        self.validation_dataset.x = None
        self.validation_dataset.y = None
        self.test_dataset.x = None
        self.test_dataset.y = None

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset.y)
        elif self.mode == 'validation':
            return len(self.validation_dataset.y)
        elif self.mode == 'test':
            return len(self.test_dataset.y)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x_item = self.train_dataset.x[idx].to(self.options.device)
            y_item = self.train_dataset.y[idx].to(self.options.device)
        elif self.mode == 'validation':
            x_item = self.validation_dataset.x[idx].to(self.options.device)
            y_item = self.validation_dataset.y[idx].to(self.options.device)
        elif self.mode == 'test':
            x_item = self.test_dataset.x[idx].to(self.options.device)
            y_item = self.test_dataset.y[idx].to(self.options.device)
        return x_item, y_item
    
    def type(self, mode='train'):
        self.mode = mode
        return self


class SklearnDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.train_dataset = getattr(skl_datasets, 'load_'+self.options.dataset_name)()

        self.test_dataset = obj()
        self.train_dataset.x, self.test_dataset.x, self.train_dataset.y, self.test_dataset.y = train_test_split(self.train_dataset.data, self.train_dataset.target, test_size=0.3, shuffle=True)
        
        self.validation_dataset = obj()
        self.train_dataset.x, self.validation_dataset.x, self.train_dataset.y, self.validation_dataset.y = train_test_split(self.train_dataset.x, self.train_dataset.y, test_size=0.2, shuffle=True)
        
        self.train_dataset.x = torch.Tensor(self.train_dataset.x)
        self.train_dataset.y = torch.Tensor(self.train_dataset.y)
        self.validation_dataset.x = torch.Tensor(self.validation_dataset.x)
        self.validation_dataset.y = torch.Tensor(self.validation_dataset.y)
        self.test_dataset.x = torch.Tensor(self.test_dataset.x)
        self.test_dataset.y = torch.Tensor(self.test_dataset.y)
        
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset.y)
        elif self.mode == 'validation':
            return len(self.validation_dataset.y)
        elif self.mode == 'test':
            return len(self.test_dataset.y)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x_item = self.train_dataset.x[idx].to(self.options.device)
            y_item = self.train_dataset.y[idx].to(self.options.device)
        elif self.mode == 'validation':
            x_item = self.validation_dataset.x[idx].to(self.options.device)
            y_item = self.validation_dataset.y[idx].to(self.options.device)
        elif self.mode == 'test':
            x_item = self.test_dataset.x[idx].to(self.options.device)
            y_item = self.test_dataset.y[idx].to(self.options.device)
        return x_item, y_item
    
    def type(self, mode='train'):
        self.mode = mode
        return self


class TorchDataset:
    def __init__(self, options):
        self.dataset = getattr(tv_datasets, options.dataset_name)(root=options.dataset_savepath, train=True, transform=transforms.ToTensor(), download=True)
        num_dataset = len(self.dataset)
        num_train = int(num_dataset*0.7)
        num_validation = num_dataset - num_train

        self.dataset = random_split(self.dataset, [num_train, num_validation])
        self.train_dataset = self.dataset[0]
        self.validation_dataset = self.dataset[1]
        self.test_dataset = getattr(tv_datasets, options.dataset_name)(root=options.dataset_savepath, train=False, transform=transforms.ToTensor(), download=True)

    def type(self, mode='train'):
        if mode == 'train':
            return self.train_dataset
        elif mode == 'validation':
            return self.validation_dataset
        elif mode == 'test':
            return self.test_dataset



def AileverDataset(options):
    torchset = ['MNIST', 'Fashion-MNIST', 'KMNIST', 'EMNIST', 'QMNIST',\
                'FakeData', 'COCO', 'Captions', 'Detection', 'LSUN',\
                'ImageNet', 'CIFAR', 'STL10', 'SVHN', 'PhotoTour',\
                'SBU', 'Flickr', 'VOC', 'Cityscapes', 'SBD',\
                'USPS', 'Kinetics-400', 'HMDB51', 'UCF101', 'CelebA']
    sklearnset = ['boston', 'breast_cancer', 'diabets', 'digits', 'iris', 'wine']
    # nlpset = [dataset.id for dataset in list_datasets()]
    nlpset = ['aeslc', 'ag_news', 'ai2_arc', 'allocine', 'anli', 'arcd', 'art',\
              'billsum', 'biomrc', 'blended_skill_talk', 'blimp', 'blog_authorship_corpus', 'bookcorpus', 'boolq', 'break_data',\
              'c4', 'cfq', 'civil_comments', 'cmrc2018', 'cnn_dailymail', 'coarse_discourse', 'com_qa', 'commonsense_qa',\
              'compguesswhat', 'coqa', 'cornell_movie_dialog', 'cos_e', 'cosmos_qa', 'crd3', 'crime_and_punish', 'csv',\
              'definite_pronoun_resolution', 'discofuse', 'docred', 'drop',\
              'eli5', 'emo', 'emotion', 'empathetic_dialogues', 'eraser_multi_rc', 'esnli', 'event2Mind',\
              'fever', 'flores', 'fquad',\
              'gap', 'germeval_14', 'ghomasHudson/cqc', 'gigaword', 'glue',\
              'hansards', 'hellaswag', 'hyperpartisan_news_detection',\
              'imdb',\
              'jeopardy', 'json',\
              'k-halid/WikiArab', 'k-halid/ar', 'kor_nli',\
              'lc_quad', 'lhoestq/c4', 'librispeech_lm', 'lince', 'lm1b', 'lordtt13/emo',\
              'math_dataset', 'math_qa', 'mlqa', 'movie_rationales', 'ms_marco', 'multi_news', 'multi_nli', 'multi_nli_mismatch', 'mwsc',\
              'natural_questions', 'newsgroup', 'newsroom',\
              'openbookqa', 'opinosis',\
              'pandas', 'para_crawl', 'pg19', 'piaf',\
              'qa4mre', 'qa_zre', 'qangaroo', 'qanta', 'qasc', 'quarel', 'quartz', 'quora', 'quoref',\
              'race', 'reclor', 'reddit', 'reddit_tifu', 'rotten_tomatoes',\
              'scan', 'scicite', 'scientific_papers', 'scifact', 'sciq', 'scitail', 'search_qa', 'sentiment140', 'snli', 'social_bias_frames',\
              'social_i_qa', 'sogou_news', 'squad', 'squad_es', 'squad_it', 'squad_v1_pt', 'squad_v2','squadshifts', 'style_change_detection', 'super_glue',\
              'ted_hrlr', 'ted_multi', 'text', 'tiny_shakespeare', 'trec', 'trivia_qa', 'tydiqa',\
              'ubuntu_dialogs_corpus',\
              'web_of_science','web_questions', 'webis/tl_dr', 'wiki40b', 'wiki_dpr', 'wiki_qa', 'wiki_snippets', 'wiki_split', 'wikihow', 'wikipedia', 'wikisql', 'wikitext',\
              'winogrande', 'wiqa', 'wmt14', 'wmt15', 'wmt16', 'wmt17', 'wmt18', 'wmt19', 'wmt_t2t', 'wnut_17',\
              'x_stance', 'xcopa', 'xnli', 'xquad', 'xsum', 'xtreme',\
              'yelp_polarity']

    if options.dataset_name in torchset:
        torchdataset = TorchDataset(options)
        train_dataset = torchdataset.type('train')
        validation_dataset = torchdataset.type('validation')
        test_dataset = torchdataset.type('test')
        return train_dataset, validation_dataset, test_dataset

    elif options.dataset_name in sklearnset:
        sklearndataset = SklearnDataset(options)
        train_dataset = sklearndataset.type('train')
        validation_dataset = sklearndataset.type('validation')
        test_dataset = sklearndataset.type('test')
        return train_dataset, validation_dataset, test_dataset

    elif options.dataset_name in nlpset:
        nlpdataset = NLPDataset(options)
        train_dataset = nlpdataset.type('train')
        validation_dataset = nlpdataset.type('validation')
        test_dataset = nlpdataset.type('test')
        return train_dataset, validation_dataset, test_dataset

def main(options):
    train_dataset, validation_dataset, test_dataset = AileverDataset(options)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    x_train, y_train = next(iter(train_dataloader))
    x_train, y_train = next(iter(validation_dataloader))
    x_train, y_train = next(iter(test_dataloader))


if __name__ == "__main__":
    options = options.load()
    main(options)
