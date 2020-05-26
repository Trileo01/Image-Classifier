import torch
from torch import nn
from torchvision import models

def load_model(arch, hidden_units):
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
        for params in model.parameters():
            params.require_grad = False

        classifier = nn.Sequential(nn.Linear(25088,hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, 568),
                                   nn.ReLU(),
                                   nn.Linear(568, 102),
                                   nn.LogSoftmax(dim = 1))
        model.classifier = classifier
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        for params in model.parameters():
            params.require_grad = False

        classifier = nn.Sequential(nn.Linear(1024,hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, 568),
                                   nn.ReLU(),
                                   nn.Linear(568, 102),
                                   nn.LogSoftmax(dim = 1))
        model.classifier = classifier

    return model