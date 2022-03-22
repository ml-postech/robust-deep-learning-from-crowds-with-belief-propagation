import torch
from torch.distributions import Dirichlet

from .utils import get_stats

def get_init_messages(ann, n_classes=2, device='cuda:0'):
    n_tasks, n_workers = get_stats(ann)
    m_tasks = torch.ones((n_workers, n_tasks, n_classes)).to(device) * (ann.transpose(0, 1) != -1).unsqueeze(dim=-1) * 0.5
    return m_tasks


def compute_marginal(ann, m_tasks, m_features=None):
    m_tasks = m_tasks + (ann.transpose(0, 1) == -1).int().unsqueeze(dim=-1)
    probs = m_tasks.prod(dim=0)
    if m_features is not None:
        probs = probs * m_features
    
    return probs


def bp_iteration(ann, m_tasks, q_prior, m_features=None, device='cuda:0', n_samples=400):
    # Update worker messages
    m_tasks_tmp = m_tasks + (ann.transpose(0, 1) == -1).int().unsqueeze(dim=-1)
    m_prods = m_tasks_tmp.prod(dim=0)
    if m_features is not None:
        m_prods = m_prods * m_features
    m_workers_new = m_prods.unsqueeze(dim=0) / m_tasks_tmp
    m_workers_new = m_workers_new.transpose(0, 1)
    m_workers_new = m_workers_new / m_workers_new.sum(dim=-1).unsqueeze(dim=-1)

    # Sample worker abilities
    q_dist = Dirichlet(q_prior.transpose(0, 1))
    qs = q_dist.sample((n_samples,)).to(device)
    torch.cuda.empty_cache()

    # Update task messages
    m_workers_sum = m_workers_new.unsqueeze(dim=0) * qs[:, :, ann].permute(0, 2, 3, 1)
    m_workers_sum = m_workers_sum.sum(dim=-1)
    m_workers_sum = m_workers_sum * (ann != -1).unsqueeze(dim=0)
    m_workers_sum = m_workers_sum + (ann == -1).int().unsqueeze(dim=0)
    m_prods = m_workers_sum.log().sum(dim=1).unsqueeze(dim=1) - m_workers_sum.log()
    m_prods = m_prods.unsqueeze(dim=-1) + qs[:, :, ann].permute(0, 2, 3, 1).log()
    m_prods = m_prods - m_prods.amax(dim=(0, 1, 3)).reshape((1, 1, m_prods.size(2), 1))
    m_tasks_new = m_prods.exp().sum(dim=0)
    m_tasks_new = m_tasks_new.transpose(0, 1)
    m_tasks_new = m_tasks_new / m_tasks_new.sum(dim=-1).unsqueeze(dim=-1)
    m_tasks_new = m_tasks_new * (ann.transpose(0, 1) != -1).unsqueeze(dim=-1)

    return m_tasks_new


def run_bp(ann, n_iters, q_prior, m_features=None, device='cuda:0', n_samples=400, n_classes=2):
    m_tasks_prev = get_init_messages(ann, n_classes, device)

    for _ in range(n_iters):
        m_tasks = bp_iteration(ann, m_tasks_prev, q_prior, m_features, device, n_samples)
        err = (m_tasks[:, :, 0] - m_tasks_prev[:, :, 0]).abs().max()
        if err < 0.03:
            break
        m_tasks_prev = m_tasks
    z_bp = compute_marginal(ann, m_tasks, m_features)
    z_bp = z_bp / z_bp.sum(dim=1).unsqueeze(dim=-1)

    return z_bp

