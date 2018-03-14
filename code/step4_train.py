
# Includes some code from:
#
#   Name: PyTorch Transfer Learning tutorial
#   License: BSD
#   Author: Sasank Chilamkurthy <https://chsasank.github.io>

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import time
import os
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import sys
import json

import pytorch_utils


data_transforms = {
    'train_RandomCrop': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        pytorch_utils.RandomVerticalFlip(),
        pytorch_utils.RandomTranspose(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_RandomResizedCrop': transforms.Compose([
        transforms.RandomResizedCrop(224, (0.4,1), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        pytorch_utils.RandomVerticalFlip(),
        pytorch_utils.RandomTranspose(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

n_gpu = torch.cuda.device_count()
VAL_INDEX = 363572

# TODO: ideally this should be moved out of global scope
base_model = None
json_path, = sys.argv[1:]
model_root = os.path.basename(json_path)[:-5]
with open(json_path) as f:
    model_desc = json.load(f)

batch_size = n_gpu * model_desc['batch_size_per_gpu']
epoch_per_decay = model_desc['epoch_factor']
n_epochs = epoch_per_decay * 3 + 2
train_transform = 'train_' + model_desc['training_augmentation']
dataset_spec = '{uncropped_size}_varsize={varsize}'.format(uncropped_size=256, varsize=model_desc['variable_size'])

if model_desc['include_val']:
    image_datasets = {
            'train': pytorch_utils.TensorDataset('training_complete_%s' % dataset_spec, data_transforms[train_transform]),
            'val': pytorch_utils.TensorDataset('training_small_%s' % dataset_spec, data_transforms['val'], min_key=VAL_INDEX),
            }
else:
    image_datasets = {
            'train': pytorch_utils.TensorDataset('training_complete_%s' % dataset_spec, data_transforms[train_transform], max_key=VAL_INDEX),
            'val': pytorch_utils.TensorDataset('training_small_%s' % dataset_spec, data_transforms['val'], min_key=VAL_INDEX),
            }

image_datasets['val'].set_class_list(image_datasets['train'].classes)
image_datasets['train'] = pytorch_utils.RandomImagePicker(image_datasets['train'])
n_classes = len(image_datasets['train'].classes)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=(x=='train'), num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

nf_dataset_sizes = {x: (~image_datasets[x].y.str.startswith('false_detection')).sum() for x in ['train', 'val']}
fake_indices = np.flatnonzero(np.array([name.startswith('false_detection') for name in class_names]))
assert len(fake_indices) == 1


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_root=None):
    assert model_root is not None

    print("Training %r" % model_root)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_nf_corrects = 0
            t0 = time.time()

            # Iterate over data.
            with tqdm(desc="epoch %d/%d: %s" % (epoch, num_epochs, phase),
                      total=len(dataloaders[phase]) * batch_size, unit="images", unit_scale=True) as progress:
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    nf_outputs = outputs.clone()
                    for i in fake_indices:
                        nf_outputs[:,i] = -np.inf
                    _, nf_preds = torch.max(nf_outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)#/32
                    running_corrects += torch.sum(preds == labels.data)
                    running_nf_corrects += torch.sum(nf_preds == labels.data)
                    progress.update(len(inputs))
                    del _
                    del nf_outputs, nf_preds, outputs, preds, loss, inputs, labels

            stats = OrderedDict()
            stats['loss'] = running_loss / dataset_sizes[phase]
            stats['acc'] = running_corrects / dataset_sizes[phase]
            stats['nf_acc'] = running_nf_corrects / nf_dataset_sizes[phase]
            stats['time'] = time.time() - t0

            print('{}  {}'.format(
                phase, '  '.join('%s: %.4f'%(k,v) for k,v in stats.items())))

        print()

    return model


def main_training(base_model=None, model_root=None):
    if base_model is None:
        from pytorch_dpn import dpn
        if model_desc['model'] == 'dpn92':
            model_ft = dpn.dpn92(pretrained=True)
        elif model_desc['model'] == 'dpn131':
            model_ft = dpn.dpn131(pretrained=True)
    else:
        model_ft = torch.load(base_model)

    fc_layer_name = list(model_ft.named_modules())[-1][0]

    assert isinstance(getattr(model_ft, fc_layer_name), nn.Conv2d)
    num_ftrs = getattr(model_ft, fc_layer_name).in_channels
    setattr(model_ft, fc_layer_name, nn.Conv2d(num_ftrs, n_classes, kernel_size=1, bias=True))

    model_ft = model_ft.cuda()

    model_ft = nn.DataParallel(model_ft)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=model_desc['lr_factor']*batch_size/128, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=epoch_per_decay, gamma=0.2)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=n_epochs, model_root=model_root)

    os.makedirs('trained_models', exist_ok=True)
    torch.save(model_ft, 'trained_models/%s.pth' % model_root)

if __name__ == '__main__':
    main_training(base_model, model_root)
