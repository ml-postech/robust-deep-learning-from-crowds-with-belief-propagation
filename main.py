import torch
import random
import argparse
import importlib
import numpy as np
from torch.utils.data import DataLoader

models = ['MV', 'BP', 'MF', 'CL', 'BayesDGC', 'deepBP', 'deepMF']


def initial_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=models, required=True)
    return parser


if __name__ == '__main__':
    parser = initial_parser()
    model_name = parser.parse_known_args()[0].model
    module = importlib.import_module(f'models.learnings.{model_name}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, required=True)
    getattr(module, 'add_args')(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('Loading data...')
    train_data = getattr(module, 'Dataset')(args, is_train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = getattr(module, 'Dataset')(args, is_train=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print('Start training!')
    model = getattr(module, 'Model')(args).to(args.device)
    getattr(module, 'train')(args, model, train_loader, test_loader) 
    
