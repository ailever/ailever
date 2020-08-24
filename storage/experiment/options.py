# built-in / external modules
import argparse

# torch
import torch

# ailever modules
from visualization import AileverVisualizer

obj = type('obj', (), {})
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
    parser.add_argument('--split_rate',
                        type=float,
                        default=0.7)

    # datasets path
    parser.add_argument('--dataset_savepath',
                        type=str,
                        default='datasets/')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='MNIST',
                        help='* [custom] \
                              internal-univariate-linear-scalar, external-univariate-linear-scalar,\
                              internal-multivariate-linear-scalar, external-multivariate-linear-scalar\
                              internal-multivariate-linear-vector, external-multivariate-linear-vector\
                              * [torchvision] \
                              MNIST, Fashion-MNIST, KMNIST, EMNIST, QMNIST, FakeData, COCO, \
                              Captions, Detection, LSUN, ImageNet, CIFAR, STL10, SVHN, PhotoTour, \
                              SBU, Flickr, VOC, Cityscapes, SBD, USPS, Kinetics-400, HMDB51, UCF101, CelebA, \
                              * [sklearn] \
                              boston, breast_cancer, diabets, digits, iris, wine, \
                              * [nlp] \
                              aeslc, ag_news, ai2_arc, allocine, anli, arcd, art, \
                              billsum, biomrc, blended_skill_talk, blimp, blog_authorship_corpus, bookcorpus, boolq, break_data, \
                              c4, cfq, civil_comments, cmrc2018, cnn_dailymail, coarse_discourse, com_qa, commonsense_qa, \
                              compguesswhat, coqa, cornell_movie_dialog, cos_e, cosmos_qa, crd3, crime_and_punish, csv, \
                              definite_pronoun_resolution, discofuse, docred, drop, \
                              eli5, emo, emotion, empathetic_dialogues, eraser_multi_rc, esnli, event2Mind, \
                              fever, flores, fquad, \
                              gap, germeval_14, ghomasHudson/cqc, gigaword, glue, \
                              hansards, hellaswag, hyperpartisan_news_detection, \
                              imdb, \
                              jeopardy, json, \
                              k-halid/WikiArab, k-halid/ar, kor_nli, \
                              lc_quad, lhoestq/c4, librispeech_lm, lince, lm1b, lordtt13/emo, \
                              math_dataset, math_qa, mlqa, movie_rationales, ms_marco, multi_news, multi_nli, multi_nli_mismatch, mwsc, \
                              natural_questions, newsgroup, newsroom, \
                              openbookqa, opinosis, \
                              pandas, para_crawl, pg19, piaf, \
                              qa4mre, qa_zre, qangaroo, qanta, qasc, quarel, quartz, quora, quoref, \
                              race, reclor, reddit, reddit_tifu, rotten_tomatoes, \
                              scan, scicite, scientific_papers, scifact, sciq, scitail, search_qa, sentiment140, snli, social_bias_frames, \
                              social_i_qa, sogou_news, squad, squad_es, squad_it, squad_v1_pt, squad_v2, squadshifts, style_change_detection, super_glue, \
                              ted_hrlr, ted_multi, text, tiny_shakespeare, trec, trivia_qa, tydiqa, \
                              ubuntu_dialogs_corpus, \
                              web_of_science, web_questions, webis/tl_dr, wiki40b, wiki_dpr, wiki_qa, wiki_snippets, wiki_split, wikihow, wikipedia, wikisql, wikitext, \
                              winogrande, wiqa, wmt14, wmt15, wmt16, wmt17, wmt18, wmt19, wmt_t2t, wnut_17, \
                              x_stance, xcopa, xnli, xquad, xsum, xtreme, \
                              yelp_polarity')
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

    # additional argument
    parser.add_argument('--add', type=obj, default=obj())
    options = parser.parse_args()
    options.add.x_train_shape = None
    options.add.y_train_shape = None
    options.add.x_validation_shape = None
    options.add.y_validation_shape = None
    options.add.x_test_shape = None
    options.add.y_test_shape = None
    options.add.vis = AileverVisualizer(options)
    return options

    
