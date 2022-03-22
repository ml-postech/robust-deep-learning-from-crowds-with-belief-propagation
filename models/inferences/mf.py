import torch
from torch import nn

from .utils import get_workers_of_task, get_tasks_of_worker, get_stats


def mf_iteration(ann, p_tasks, prior):
    softmax = nn.Softmax(dim=1)
    n_tasks, n_workers = get_stats(ann)

    qs = torch.FloatTensor([prior for _ in range(n_workers)]) 

    for u in range(n_workers):
        tasks_of_worker = get_tasks_of_worker(ann, u)

        for t in tasks_of_worker:
            qs[u, :, ann[t, u]] += p_tasks[t]

    for t in range(n_tasks):
        workers_of_task = get_workers_of_task(ann, t)

        tmp = -qs[workers_of_task].sum(dim=2).digamma().sum(dim=0)
        for u in workers_of_task:
            tmp += qs[u, :, ann[t, u]].digamma()
        p_tasks[t] = tmp

    p_tasks = softmax(p_tasks)
    return p_tasks


def run_mf(ann, n_iters, prior, n_classes=2):
    n_tasks, _ = get_stats(ann)
    p_tasks = torch.ones(n_tasks, n_classes) * 0.5

    for _ in range(n_iters):
        p_tasks = mf_iteration(ann, p_tasks, prior)

    return p_tasks

