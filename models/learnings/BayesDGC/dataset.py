from ..base import BaseDataset, add_base_dataset_args


def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset')
    add_base_dataset_args(group)


class Dataset(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)
