import torch
from torch import nn
import torch.nn.functional as F

from ..base import Classifier
from ...inferences import run_bp

def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--given_alpha', type=float, default=2.)
    group.add_argument('--given_beta', type=float, default=1.)
    group.add_argument('--clipping', type=float, default=1.0)
    group.add_argument('--n_iters', type=int, default=100)
    group.add_argument('--n_samples', type=int, default=400)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.n_classes = 2
        self.given_alpha = args.given_alpha
        self.given_beta = args.given_beta
        self.clipping = args.clipping
        self.n_iters = args.n_iters
        self.n_samples = args.n_samples
        self.classifier = Classifier()

        self.prior = torch.FloatTensor([[self.given_alpha, self.given_beta], [self.given_beta, self.given_alpha]])

    def inference(self, x, ann):
        z_learning = F.softmax(self.classifier(x), dim=-1)
        z_learning = torch.clamp(z_learning, min=1 - self.clipping, max=self.clipping) + 1e-9

        with torch.no_grad():
            z_inference = run_bp(ann, self.n_iters, self.prior, z_learning, x.device, self.n_samples)

        return z_inference, z_learning

    def forward(self, x, ann):
        z_inference, z_learning = self.inference(x, ann)
        loss = -(z_inference * z_learning.log()).sum()
        return loss

