import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset


def add_base_dataset_args(parser):
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--blur', type=int)
    parser.add_argument('--n_tasks', type=int, default=1000)
    parser.add_argument('--true_alpha', type=float, default=2.)
    parser.add_argument('--true_beta', type=float, default=1.)
    parser.add_argument('--n_tasks_per_worker', type=int, default=4)
    parser.add_argument('--n_workers_per_task', type=int, default=3)
    parser.add_argument('--n_extreme_spammers', type=int, default=0)


class BaseDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train

        root_dir = Path(args.data_path) / ('train' if self.is_train else 'test')
        self.images = np.load(root_dir / 'data.npy')
        self.labels = np.load(root_dir / 'labels.npy')
        if self.is_train:
            self.images = self.images[86:86 + args.n_tasks]
            self.labels = self.labels[86:86 + args.n_tasks]

        if args.blur is not None:
            blur = ImageFilter.GaussianBlur(args.blur)
            for idx in range(self.images.shape[0]):
                img = Image.fromarray(self.images[idx])
                img = img.filter(blur)
                self.images[idx] = np.array(img)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4880633112757274, 0.45490764498668607, 0.45490764498668607), 
                (0.26311969928021317, 0.25659017730169825, 0.259108662261481)
            )
        ])

        if self.is_train:
            ann, _ = generate_labels(self.labels, args.true_alpha, args.true_beta, args.n_tasks_per_worker, args.n_workers_per_task, args.n_extreme_spammers)
            self.annotations = self.preprocess(ann)

    def __len__(self):
        return self.images.shape[0]

    def preprocess(self, ann):
        return ann

    def __getitem__(self, idx):
        x = self.images[idx]
        x = self.transforms(x)
        y = self.labels[idx]

        if self.is_train:
            ann = self.annotations[idx]
            return x, y, ann
        else:
            return x, y


def generate_labels(ground_truth, alpha, beta, n_tasks_per_worker, n_workers_per_task, n_extreme_spammers=0):
    n_tasks = ground_truth.shape[0]
    solved = [0] * n_tasks
    n_workers = int(n_tasks * n_workers_per_task / n_tasks_per_worker)
    n_classes = 2

    labels = []
    new_qs = []

    for _ in range(n_extreme_spammers):
        worker_labels = random.choices([0, 1], k=n_tasks)
        labels.append(worker_labels)

    dist = torch.distributions.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta]))
    for _ in range(n_workers):
        q = dist.sample().numpy()[0]
        q = [[q, 1 - q], [1 - q, q]]
        new_qs.append(q)

        worker_labels = [-1] * n_tasks
        task_indices = list(range(n_tasks))
        random.shuffle(task_indices)

        cnt = 0
        for t in task_indices:
            if solved[t] == n_workers_per_task:
                continue
            worker_labels[t] = np.random.choice(list(range(n_classes)), p=q[ground_truth[t]])
            cnt += 1
            solved[t] += 1

            if cnt == n_tasks_per_worker:
                break
        if cnt != 0:
            labels.append(worker_labels)
    
    labels = np.array(labels).transpose()
    return labels, new_qs


