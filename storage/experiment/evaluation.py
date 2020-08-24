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
    _, _, test_dataset = AileverDataset(options)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=True, drop_last=False)

    model = AileverModel(options)
    model.load_state_dict(torch.load(f'.Log/model_{options.id}.pth'));          print(f'[AILEVER] The file ".Log/model_{options.id}.pth" is successfully loaded!')
    model = model.to(options.device)
    
    predictions = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (x_train, y_train) in enumerate(test_dataloader):
            x_train = x_train.view(options.batch_size, 28*28).to(options.device)
            y_train = y_train.to(options.device)
            
            time_start = time.time()
            hypothesis = model(x_train)
            time_end = time.time()
            for i in range(len(y_train)):
                html = f"""<b>{options.dataset_name.upper()}</b><br>
                           <b>[EVALUATION][{i+1}/{batch_idx+1}/{len(test_dataloader)}]</b> <br>
                           * SIZE : {x_train.size()} <br>
                           * TURE : {y_train[i].data} <br>
                           * PRED : {torch.argmax(hypothesis[i], dim=-1).data} <br>
                           * TIME : {time_end - time_start:.10f}(sec)"""
                options.add.vis.texting(html=html)
                predictions[f'batch index{batch_idx}{i}'] = {'input data':x_train[i].tolist(), 'prediction':torch.argmax(hypothesis[i], dim=-1).tolist(), 'true':y_train[i].tolist()}
        print(f'- Prediction/True : {torch.argmax(hypothesis[0], dim=-1).data}/{y_train[0].data}')
    

    if not os.path.isdir('evaluations') : os.mkdir('evaluations')
    json.dump(predictions, open(f'evaluations/{options.id}.json', 'w'))
    print(f'[AILEVER] The file "evaluations/{options.id}.json" is successfully saved!')

if __name__ == "__main__":
    options = options.load()
    evaluation(options)
