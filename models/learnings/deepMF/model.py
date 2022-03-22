import torch
from torch import nn
import torch.nn.functional as F

from ..base import Classifier


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--given_alpha', type=float, default=2.)
    group.add_argument('--given_beta', type=float, default=1.)
    group.add_argument('--clipping', type=float, default=1.0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.n_classes = 2
        self.given_alpha = args.given_alpha
        self.given_beta = args.given_beta
        self.clipping = args.clipping
        self.n_worker = args.n_tasks * args.n_workers_per_task // args.n_tasks_per_worker
        self.classifier = Classifier()
        self.criterion = nn.KLDivLoss(reduction='sum')

        prior = [[self.given_alpha, self.given_beta], [self.given_beta, self.given_alpha]]
        self.confusion = torch.FloatTensor([prior for _ in range(self.n_worker)])

    def inference(self, x, ann):
        z_learning = F.softmax(self.classifier(x), dim=-1)
        z_learning = torch.clamp(z_learning, min=1 - self.clipping, max=self.clipping) + 1e-9
        z_learning = z_learning.cpu()

        with torch.no_grad():
            z_inference = z_learning.log()

        for t in range(x.size(0)):
            workers_of_task = (ann[t].view(-1) != -1).nonzero().view(-1)

            tmp = -self.confusion[workers_of_task].sum(dim=2).digamma().sum(dim=0)
            for u in workers_of_task:
                tmp += self.confusion[u, :, ann[t, u]].digamma()
            z_inference[t] += tmp

        z_inference = F.softmax(z_inference, dim=-1)
        return z_inference, z_learning

    def forward(self, x, ann):
        z_inference, z_learning = self.inference(x, ann)
        
        for u in range(ann.size(1)):
            tasks_of_worker = (ann[:, u].view(-1) != -1).nonzero().view(-1)

            self.confusion[u] = torch.FloatTensor([[self.given_alpha, self.given_beta], [self.given_beta, self.given_alpha]])
            for t in tasks_of_worker:
                self.confusion[u, :, ann[t, u]] += z_inference[t]

        loss = self.criterion(z_learning.log(), z_inference)
        return loss

