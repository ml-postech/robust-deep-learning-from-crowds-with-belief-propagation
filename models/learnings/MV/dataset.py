import numpy as np

from ..base import BaseDataset, add_base_dataset_args


def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset')
    add_base_dataset_args(group)


class Dataset(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

    def preprocess(self, ann):
        labels_mv = np.zeros(self.args.n_tasks)
        for t in range(ann.shape[0]):
            cnt = np.array([0, 0])
            for label in ann[t]:
                if label != -1:
                    cnt[label] += 1
            labels_mv[t] = cnt.argmax(axis=0)

        labels_mv = labels_mv.astype(np.int64)
        return labels_mv

