import torch
from ...inferences import run_bp

from ..base import BaseDataset, add_base_dataset_args


def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset')
    add_base_dataset_args(group)
    group.add_argument('--given_alpha', type=float, default=2.)
    group.add_argument('--given_beta', type=float, default=1.)
    group.add_argument('--n_iters', type=int, default=100)
    group.add_argument('--n_samples', type=int, default=400)


class Dataset(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

    def preprocess(self, ann):
        ann = torch.LongTensor(ann).to(self.args.device)
        prior = torch.FloatTensor([
            [self.args.given_alpha, self.args.given_beta],
            [self.args.given_beta, self.args.given_alpha]
        ])

        labels_bp = run_bp(ann, self.args.n_iters, prior, device=self.args.device, n_samples=self.args.n_samples)
        labels_bp = labels_bp.cpu().numpy()
        return labels_bp

