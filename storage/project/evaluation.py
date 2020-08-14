# built-in / external modules
import os
import json
import time

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ailever modules
from dataset import AileverDataset
from models import AileverModel
import options


def evaluation(options):
    dataset = AileverDataset(options)
    test_dataloader = DataLoader(dataset.type('test'), batch_size=options.batch_size, shuffle=True, drop_last=False)

    model = AileverModel(options)
    model.load_state_dict(torch.load(f'.Log/model_{options.id}.pth'));          print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully loaded!')
    model = model.to(options.device)
    
    predictions = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (x_train, y_train) in enumerate(test_dataloader):
            time_start = time.time()
            hypothesis = model(x_train)
            time_end = time.time()

            for i in range(len(y_train)):
                html = f"""<b>[EVALUATION][{i+1}/{batch_idx+1}/{len(test_dataloader)}]</b> <br>
                           * SIZE : {x_train.size()} <br>
                           * INPUT : {x_train[i].data} <br>
                           * TURE : {y_train[i].data} <br>
                           * PRED : {hypothesis[i].data} <br>
                           * TIME : {time_end - time_start:.10f}(sec)"""
                options.add.vis.texting(html=html)
                predictions[f'batch index{batch_idx}{i}'] = {'input data':x_train[i].tolist(), 'prediction':hypothesis[i].tolist(), 'true':y_train[i].tolist()}
        print(f'- Prediction : {x_train[0].data} >> {hypothesis[0].data}')
    

    if not os.path.isdir('evaluation') : os.mkdir('evaluation')
    json.dump(predictions, open(f'evaluation/{options.id}.json', 'w'))
    print(f'[AILEVER] The file "evaluation/{options.id}.json" is successfully saved!')

if __name__ == "__main__":
    options = options.load()
    evaluation(options)
