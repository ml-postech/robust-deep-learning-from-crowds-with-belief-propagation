from .model import add_model_args
from .dataset import add_dataset_args
from .train import add_train_args


def add_args(parser):
    add_train_args(parser)
    add_model_args(parser)
    add_dataset_args(parser)

