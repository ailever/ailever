# built-in / external modules
import os
import json

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ailever modules
from datasets import AileverDataset
from models import AileverModel
import options


def evaluation(options):
    dataset = AileverDataset(options)
    test_dataloader = DataLoader(dataset.type('test'), batch_size=options.batch_size, shuffle=True)

    model = AileverModel(options)
    model.load_state_dict(torch.load(f'.Log/model_{options.id}.pth'));          print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully loaded!')
    model = model.to(options.device)
    
    predictions = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (x_train, y_train) in enumerate(test_dataloader):
            hypothesis = model(x_train)
            for i in range(options.batch_size):
                html = f'<b>[EVALUATION]</b> <br>* SIZE : {x_train.size()} <br>* INPUT : {x_train[i].data} <br>* TURE : {y_train[i].data} <br>* PRED : {hypothesis[i].data}'
                options.vis.texting(html=html)
                predictions[f'batch index{batch_idx}{i}'] = {'input data':x_train[i].tolist(), 'prediction':hypothesis[i].tolist(), 'true':y_train[i].tolist()}
        print(f'- Prediction : {x_train[0].data} >> {hypothesis[0].data}')
    

    if not os.path.isdir('evaluation') : os.mkdir('evaluation')
    json.dump(predictions, open(f'evaluation/prediction_{options.id}.json', 'w'))
    print(f'[AILEVER] The file "evaluation/prediction_{options.id}.json" is successfully saved!')

if __name__ == "__main__":
    options = options.load()
    evaluation(options)
