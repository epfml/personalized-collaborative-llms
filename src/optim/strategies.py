import json
from argparse import Namespace
from contextlib import nullcontext
from typing import List, Dict, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data.agnews import MIXED_TOKENS_DIST as AG_MIXED_TOKENS_DIST, SPECIFIC_TOKENS_DIST as AGNEWS_SPECIFIC_TOKENS_DIST
from data.github_wiki import MIXED_TOKENS_DIST as GIT_MIXED_TOKENS_DIST, SPECIFIC_TOKENS_DIST as GIT_SPECIFIC_TOKENS_DIST
from data.three_multi import MIXED_TOKENS_DIST as THREE_MULTI_MIXED_TOKENS_DIST, SPECIFIC_TOKENS_DIST as THREE_MULTI_SPECIFIC_TOKENS_DIST
from optim.utils import get_batch, eval


def aggregate(clients: List[List[nn.Module | Optimizer | LRScheduler]], trust: str,
              data: Dict[str, List[np.ndarray] | np.ndarray], sequence_length: int, batch_size: int,
              type_ctx: Union[nullcontext, autocast], extra_args: Namespace) -> None:
    trust_weights = None
    if trust == 'none':
        return
    elif trust == 'naive':
        trust_weights = __naive_weights(clients)
    elif trust == 'static':
        trust_weights = __static_weights(clients, extra_args.dataset)

    elif 'dynamic' in trust:
        trust_weights = __trust_weights_weights(clients, similarity_weights)
        if trust == 'dynamic':
            trust_weights = F.softmax(trust_weights, dim=1)
        elif trust == 'dynamic-thresh':
            trust_weights = __threshold(trust_weights, 0.5)
        elif trust == 'dynamic-top-k':
            trust_weights = __top_k(trust_weights, extra_args.k)

    elif 'ref' in trust:
        trust_weights = __trust_weights_ref(clients, data, sequence_length, batch_size, type_ctx, extra_args)
        if trust == 'dynamic-ref':
            trust_weights = F.softmax(trust_weights, dim=1)
        elif trust == 'dynamic-thresh-ref':
            trust_weights = __threshold(trust_weights, -30)
        elif trust == 'dynamic-top-k-ref':
            trust_weights = __top_k(trust_weights, extra_args.k)
    elif 'token' in trust:
        trust_weights = __trust_weights_token(clients, data, sequence_length, batch_size, type_ctx, extra_args)
        if trust == 'dynamic-token':
            trust_weights = F.softmax(trust_weights, dim=1)
        elif trust == 'dynamic-thresh-token':
            trust_weights = __threshold(trust_weights, -50)
        elif trust == 'dynamic-top-k-token':
            trust_weights = __top_k(trust_weights, extra_args.k)

    __weighted_average(clients, trust_weights, extra_args)


def __threshold(tensor: Tensor, threshold: float) -> Tensor:
    top_k_values, top_k_indices = torch.topk(tensor, 2, dim=-1)
    tensor[tensor <= threshold] = -1e9
    tensor.scatter_(-1, top_k_indices, top_k_values)
    return F.softmax(tensor, dim=1)


def __top_k(tensor: Tensor, top_k: int) -> Tensor:
    top_k_values, top_k_indices = torch.topk(tensor, top_k, dim=-1)
    mask = torch.zeros_like(tensor)
    mask = torch.fill(mask, -1e9)
    mask.scatter_(-1, top_k_indices, top_k_values)
    return F.softmax(mask, dim=1)


def __weighted_average(clients: List[List[nn.Module | Optimizer | LRScheduler]], trust_weights: Tensor,
                       extra_args: Namespace) -> None:
    if extra_args.wandb:
        wandb.log({'Trust weights': json.dumps(np.array(trust_weights).tolist())}, commit=False)

    weights = {}
    for id, client in enumerate(clients):
        for name, param in client[0].named_parameters():
            if param.requires_grad:
                if name in weights:
                    weights[name][id] = param.data.clone()
                else:
                    weights[name] = {}
                    weights[name][id] = param.data.clone()

    for idx, client in enumerate(clients):
        model, _, _ = client

        for name, param in model.named_parameters():
            if param.requires_grad:
                val = torch.zeros_like(param)
                for i in range(len(clients)):
                    val += trust_weights[idx, i] * weights[name][i]
                param.data = val

    del weights


def __naive_weights(clients: List[List[nn.Module | Optimizer | LRScheduler]]) -> Tensor:
    trust_weights = torch.zeros((len(clients), len(clients)))
    return torch.fill(trust_weights, 1 / len(clients))


def __static_weights(clients: List[List[nn.Module | Optimizer | LRScheduler]], dataset: str) -> Tensor:
    if dataset == 'agnews_mixed':
        TOKENS_DIST = AG_MIXED_TOKENS_DIST
    elif dataset == 'agnews_specific':
        TOKENS_DIST = AGNEWS_SPECIFIC_TOKENS_DIST
    elif dataset == 'three_multi_specific':
        TOKENS_DIST = THREE_MULTI_SPECIFIC_TOKENS_DIST
    elif dataset == 'three_multi_mixed':
        TOKENS_DIST = THREE_MULTI_MIXED_TOKENS_DIST
    elif dataset == 'github_wiki_specific':
        TOKENS_DIST = GIT_SPECIFIC_TOKENS_DIST
    elif dataset == 'github_wiki_mixed':
        TOKENS_DIST = GIT_MIXED_TOKENS_DIST
    TOKENS_DIST = torch.tensor(TOKENS_DIST)
    trust_weights = TOKENS_DIST @ TOKENS_DIST.T
    trust_weights = trust_weights.repeat(len(clients) // TOKENS_DIST.size(0), len(clients) // TOKENS_DIST.size(0))
    trust_weights /= (len(clients) // TOKENS_DIST.size(0))
    return trust_weights[:len(clients), :len(clients)]


def similarity_weights(client1: nn.Module, client2: nn.Module,
                       similarity: Callable[[Tensor, Tensor], Tensor] = F.cosine_similarity) -> float:
    score = 0
    total_size = 0
    for (name1, param1), (name2, param2) in zip(client1.named_parameters(), client2.named_parameters()):
        if name1 != name2:
            raise NameError(f'Should be the same: {name1} != {name2}')
        if param1.requires_grad:
            sim = similarity(param1, param2)
            total_size += sim.size(0)
            score += torch.sum(sim).detach().item()

    return score / total_size


def __trust_weights_weights(clients: List[List[nn.Module | Optimizer | LRScheduler]],
                            sim_func: Callable[[nn.Module, nn.Module, Callable[[Tensor, Tensor], Tensor]], float],
                            similarity: Callable[[Tensor, Tensor], Tensor] = F.cosine_similarity) -> Tensor:
    trust_weight = torch.zeros((len(clients), len(clients)))
    for idx1, (model1, _, _) in enumerate(clients):
        for idx2, (model2, _, _) in enumerate(clients):
            if idx2 <= idx1:
                score = sim_func(model1, model2, similarity)
                trust_weight[idx1, idx2] = score
                trust_weight[idx2, idx1] = score
    return trust_weight


def __trust_weights_ref(clients: List[List[nn.Module | Optimizer | LRScheduler]],
                        data: Dict[str, List[np.ndarray] | np.ndarray], sequence_length: int, batch_size: int,
                        type_ctx: Union[nullcontext, autocast], extra_args: Namespace) -> Tensor:
    trust_weights = torch.zeros((len(clients), len(clients)))
    for model, _, _ in clients:
        model.eval()
    for id1 in range(len(clients)):
        for id2, (model, _, _) in enumerate(clients):
            model.eval()
            _, _, val_perplexity = eval(model, data['val'][id1], sequence_length, batch_size,
                                        extra_args.device, max_num_batches=12, ctx=type_ctx)
            trust_weights[id1, id2] = val_perplexity
            model.train()
    for model, _, _ in clients:
        model.train()
    return -trust_weights


def __trust_weights_token(clients: List[List[nn.Module | Optimizer | LRScheduler]],
                          data: Dict[str, List[np.ndarray] | np.ndarray], sequence_length: int, batch_size: int,
                          type_ctx: Union[nullcontext, autocast], extra_args: Namespace) -> Tensor:
    logits = [[] for _ in range(len(clients))]
    for model, _, _ in clients:
        model.eval()
    for j in range(4):
        print(f'\r{j} batch ref', end='')
        x, y = get_batch(data['ref'], sequence_length, batch_size, extra_args.device)
        for id, (model, _, _) in enumerate(clients):
            with type_ctx:
                outputs = model(x, get_logits=True)
            logits_out = outputs['logits'].detach()
            v, _ = torch.topk(logits_out, 100)
            logits_out[logits_out < v[:, :, [-1]]] = 0
            logits_out = logits_out.to_sparse_coo()
            logits[id].append(logits_out)
    for model, _, _ in clients:
        model.train()

    res = torch.zeros((len(clients), len(clients)))
    for id1 in range(len(clients)):
        for id2 in range(len(clients)):
            sim = 0
            for j in range(4):
                sim += torch.sum(torch.abs(logits[id1][j] - logits[id2][j])).item()
            res[id1, id2] = sim / (4 * batch_size)

    res = F.normalize(res, p=1, dim=1)
    return -res * 10
