def get_stats(ann):
    n_tasks = ann.size(0)
    n_workers = ann.size(1)
    return n_tasks, n_workers


def get_tasks_of_worker(ann, u):
    return (ann[:, u].view(-1) != -1).nonzero().view(-1)


def get_workers_of_task(ann, t):
    return (ann[t].view(-1) != -1).nonzero().view(-1)

