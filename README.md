# Robust Deep Learning from Crowds with Belief Propagation
This repository is the official implementation of ["Robust Deep Learning from Crowds with Belief Propagation"](https://arxiv.org/abs/2111.00734) accepted by AISTATS 2022.

## Abstract
Crowdsourcing systems enable us to collect large-scale dataset, but inherently suffer from noisy labels of low-paid workers. We address the inference and learning problems using such a crowdsourced dataset with noise. Due to the nature of sparsity in crowdsourcing, it is critical to exploit both probabilistic model to capture worker prior and neural network to extract task feature despite risks from wrong prior and overfitted feature in practice. We hence establish a neural-powered Bayesian framework, from which we devise deepMF and deepBP with different choice of variational approximation methods, mean field (MF) and belief propagation (BP), respectively. This provides a unified view of existing methods, which are special cases of deepMF with different priors. In addition, our empirical study suggests that deepBP is a new approach, which is more robust against wrong prior, feature overfitting and extreme workers thanks to the more sophisticated BP than MF.

## Usages
You can reproduce the experiments from our paper using the following command:
```
> python main.py --model deepBP --help
usage: main.py [-h] --model {MV,BP,MF,CL,BayesDGC,deepBP,deepMF} [--seed SEED] [--n_epochs N_EPOCHS] [--test_interval TEST_INTERVAL] [--device DEVICE] [--batch_size BATCH_SIZE]
               [--lr LR] [--given_alpha GIVEN_ALPHA] [--given_beta GIVEN_BETA] [--clipping CLIPPING] [--n_iters N_ITERS] [--n_samples N_SAMPLES] --data_path DATA_PATH
               [--blur BLUR] [--n_tasks N_TASKS] [--true_alpha TRUE_ALPHA] [--true_beta TRUE_BETA] [--n_tasks_per_worker N_TASKS_PER_WORKER]
               [--n_workers_per_task N_WORKERS_PER_TASK] [--n_extreme_spammers N_EXTREME_SPAMMERS]

optional arguments:
  -h, --help            show this help message and exit
  --model {MV,BP,MF,CL,BayesDGC,deepBP,deepMF}

Training:
  --seed SEED
  --n_epochs N_EPOCHS
  --test_interval TEST_INTERVAL
  --device DEVICE
  --batch_size BATCH_SIZE
  --lr LR

Model:
  --given_alpha GIVEN_ALPHA
  --given_beta GIVEN_BETA
  --clipping CLIPPING
  --n_iters N_ITERS
  --n_samples N_SAMPLES

Dataset:
  --data_path DATA_PATH
  --blur BLUR
  --n_tasks N_TASKS
  --true_alpha TRUE_ALPHA
  --true_beta TRUE_BETA
  --n_tasks_per_worker N_TASKS_PER_WORKER
  --n_workers_per_task N_WORKERS_PER_TASK
  --n_extreme_spammers N_EXTREME_SPAMMERS
```
Because the hyperparameters differs for each model, you should check the hyperparameters for your desired model using the above command.

### Models
For the models, we implemented `MV`, `BP`, `MF`, `CL`, `BayesDGC` and our methods `deepBP` and `deepMF`.
- `MV`: Learn a classifier with majority-voted labels.
- `BP` and `MF`: Learn a classifier with the labels obtained by denoising the crowdsourced data using BP and MF [(Liu et al., 2012)](https://papers.nips.cc/paper/2012/hash/cd00692c3bfe59267d5ecfac5310286c-Abstract.html).
- `CL`: Implementation of [Rodrigues and Pereira, 2018](https://arxiv.org/abs/1709.01779) with a regularizer proposed in [Tanno et al. 2019](https://arxiv.org/abs/1902.03680).
- `BayesDGC`: Implementation of [Li et al., 2021](http://scis.scichina.com/en/2021/130104.pdf).
- `deepBP` and `deepMF`: Implementation of our methods.

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{ho2022crowds,
  title={Robust Deep Learning from Crowds with Belief Propagation},
  author={Hoyoung Kim* and Seunghyuk Cho* and Dongwoo Kim and Jungseul Ok},
  booktitle=AISTATS,
  year={2022},
  url={https://arxiv.org/abs/2111.00734}
}
```
