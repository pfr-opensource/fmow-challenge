import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
import sys
import pandas as pd
from tqdm import tqdm
import json

import fmow_helper
import pytorch_utils

def get_dataset(variable_size, seed):
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    pre_dataset = pytorch_utils.TensorDataset('test_complete_256_varsize=%d' % variable_size)
    dataset = pytorch_utils.MultiCrop(pre_dataset, 'crop10a', data_transforms['test'], seed)
    return dataset

def save_prediction(model, path, dataset, batch_size):
    """
    Run the forward pass of the model, and save the prediction to path in CSV format.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.train(False)

    pred = []
    with tqdm(desc="predict %r" % path, unit="images", unit_scale=True, total=len(dataset) // dataset.num_crops) as progress:
        for inputs, labels in dataloader:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            outputs = model(inputs)
            pred.append(outputs.data.cpu().numpy())
            progress.update(len(inputs) / dataset.num_crops)
            del outputs, inputs, labels

    pred = np.concatenate(pred)
    classes = fmow_helper.NEW_MODEL_CATEGORIES
    df = pd.DataFrame(pred, dataset.indices, classes)
    df = df.groupby(df.index.values).mean()
    df.to_csv(path)

def save_prediction_from_path(json_path):
    """
    Read the JSON description of the model at the specified path, make the prediction, and save it.
    """
    with open(json_path) as f:
        model_desc = json.load(f)

    model_path = 'trained_models/' + os.path.basename(json_path[:-5]) + '.pth'
    model_root = os.path.basename(json_path)[:-5]

    seed = model_desc['seed']
    variable_size = model_desc['variable_size']

    model = torch.load(model_path).module.cuda()
    model = nn.DataParallel(model)
    n_gpu = len(model.device_ids)

    batch_size = model_desc['batch_size_per_gpu'] * n_gpu

    os.makedirs('working/single_model_prediction', exist_ok=True)
    save_prediction(model, 'working/single_model_prediction/%s.csv' % model_root, get_dataset(variable_size, seed), batch_size)

def main():
    args = sys.argv[1:]
    if not args:
        sys.exit("no model specified")

    for json_path in args:
        save_prediction_from_path(json_path)

if __name__=='__main__':
    main()
