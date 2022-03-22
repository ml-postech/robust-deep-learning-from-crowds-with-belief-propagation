import torch
from torch import nn
import torch.nn.functional as F

from ..base import Classifier


def add_model_args(parser):
    parser.add_argument_group('Model')


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.classifier = Classifier()
        self.criterion = nn.KLDivLoss(reduction='sum')


    def forward(self, x, z_bp):
        z_pred = F.softmax(self.classifier(x), dim=-1)
        loss = self.criterion(z_pred.log(), z_bp)

        return loss

