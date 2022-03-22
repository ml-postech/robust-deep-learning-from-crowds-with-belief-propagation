import torch
from torch import nn
import torch.nn.functional as F

from ..base import Classifier


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--init', type=float, default=1.)
    group.add_argument('--lam', type=float, default=0.)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.n_classes = 2
        self.init = args.init
        self.lam = args.lam
        self.n_worker = args.n_tasks * args.n_workers_per_task // args.n_tasks_per_worker
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

        w_weight = [torch.eye(self.n_classes) * self.init for _ in range(self.n_worker)]
        self.confusion = nn.parameter.Parameter(torch.stack(w_weight), requires_grad=True)

    def forward(self, x, ann):
        z_pred = F.softmax(self.classifier(x), dim=1)
        pm = F.softmax(self.confusion, dim=2)
        ann_pred = torch.einsum('ik,jkl->ijl', z_pred, pm).view((-1, self.n_classes))

        reg = torch.zeros(1, device=x.device)
        for i in range(self.n_worker):
            reg += pm[i, 0, 0].log()

        ann = ann.view(-1)
        loss = self.criterion(ann_pred, ann) + self.lam * reg

        return loss

