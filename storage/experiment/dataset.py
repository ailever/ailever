# built-in / external modules
import json
import pickle
import h5py
import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader

# ailever modules
import options
from datasets import *


def AileverDataset(options):
    """ List of datasets for machine-learning research
    https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
    """
    customset = ['internal-univariate-linear-scalar', 'external-univariate-linear-scalar',\
                 'internal-multivariate-linear-scalar', 'external-multivariate-linear-scalar',\
                 'internal-multivariate-linear-vector', 'external-multivariate-linear-vector']
    timeseriesset = ['independent-univariate-unistep', 'independent-univariate-multistep', \
                     'independent-multivariate-unistep', 'independent-multivariate-multistep', \
                     'dependent-univariate-unistep', 'dependent-univariate-multistep', \
                     'dependent-multivariate-unistep', 'dependent-multivariate-multistep']
    torchvisionset = ['MNIST', 'Fashion-MNIST', 'KMNIST', 'EMNIST', 'QMNIST',\
                      'FakeData', 'COCO', 'Captions', 'Detection', 'LSUN',\
                      'ImageNet', 'CIFAR10', 'CIFAR100', 'STL10', 'SVHN', 'PhotoTour',\
                      'SBU', 'Flickr', 'VOC', 'Cityscapes', 'SBD',\
                      'USPS', 'Kinetics-400', 'HMDB51', 'UCF101', 'CelebA']
    sklearnset = ['boston', 'breast_cancer', 'diabets', 'digits', 'iris', 'wine']
    # nlpset = [dataset.id for dataset in nlp.list_datasets()]
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
    

    print(f'[AILEVER] The dataset "{options.dataset_name}" have been loaded!')
    if options.dataset_name in customset:
        customdataset = CustomDataset(options)
        train_dataset = customdataset.type('train')
        validation_dataset = customdataset.type('validation')
        test_dataset = customdataset.type('test')
        return train_dataset, validation_dataset, test_dataset
    
    elif options.dataset_name in torchvisionset:
        torchvisiondataset = TorchVisionDataset(options)
        train_dataset = torchvisiondataset.type('train')
        validation_dataset = torchvisiondataset.type('validation')
        test_dataset = torchvisiondataset.type('test')
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
 
    elif options.dataset_name in timeseriesset:
        timeseriesdataset = TimeSeriesDataset(options)
        train_dataset = timeseriesdataset.type('train')
        validation_dataset = timeseriesdataset.type('validation')
        test_dataset = timeseriesdataset.type('test')
        return train_dataset, validation_dataset, test_dataset

    else:
        raise Exception(f"Sorry, it failed to load dataset {options.dataset_name}")

def main(options):
    train_dataset, validation_dataset, test_dataset = AileverDataset(options)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    x_train, y_train = next(iter(train_dataloader))
    print(x_train, y_train)
    x_validation, y_validation = next(iter(validation_dataloader))
    print(x_validation, y_validation)
    x_test, y_test = next(iter(test_dataloader))
    print(x_test, y_test)
    

if __name__ == "__main__":
    options = options.load()
    options.dataset_name = 'internal-univariate-linear-scalar'
    options.dataset_name = 'MNIST'
    main(options)
