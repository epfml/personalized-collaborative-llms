import time
from argparse import Namespace
from contextlib import nullcontext
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .strategies import aggregate
from .utils import eval, get_batch
from distributed.ddp import DataParallelDistributedBackend
from distributed.single import SingleNodeBackend


def train_lora(clients: List[List[nn.Module | Optimizer | LRScheduler]], data: Dict[str, List[np.ndarray]],
               iterations: int, acc_steps: int, batch_size: int, sequence_length: int, eval_freq: int,
               distributed_backend: Union[DataParallelDistributedBackend, SingleNodeBackend],
               extra_args: Namespace) -> Dict[str, List[List[float]]]:
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)

    num_clients = len(clients)
    itr, substep, best_val_loss, text_table = [0] * num_clients, [0] * num_clients, [
        float('inf')] * num_clients, None  # best_val_loss not used atm, early stopping not recommended but possible

    stats = {'train_loss': [[] for _ in range(num_clients)], 'val_loss': [[] for _ in range(num_clients)],
             'val_pp': [[] for _ in range(num_clients)], 'val_acc': [[] for _ in range(num_clients)]}

    num_substeps_per_epoch = []
    for i in range(num_clients):
        num_substeps_per_epoch.append(len(data['train'][i]) // (batch_size * sequence_length))

    if not extra_args.no_compile:
        print(f'Compiling model ...')
        for i in range(num_clients):
            clients[i][0] = torch.compile(clients[i][0], dynamic=True)  # requires pytorch 2.0+

    for i in range(num_clients):
        clients[i][0].train()

    t0 = time.time()
    while itr[-1] < iterations:
        for i in range(num_clients):
            print(f'\r{i} {itr[i]}', end='')
            model, opt, scheduler = clients[i]

            for microstep_idx in range(acc_steps):  # gradient accumulation
                x, y = get_batch(data['train'][i], sequence_length, batch_size, device=extra_args.device)
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx,
                                                                               gradient_accumulation_steps=acc_steps):
                        outputs = model(x, targets=y)

                loss = outputs['loss'] / acc_steps
                loss.backward()
                substep[i] += 1

            if extra_args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)
            opt.step()
            scheduler.step()
            itr[i] += 1

        # aggregate models
        if itr[-1] % extra_args.trust_freq == 0 and itr[-1] >= extra_args.pretraining_rounds - 1:
            if extra_args.trust == 'none':
                pass
            elif extra_args.trust == 'naive':
                __average(clients)
            elif extra_args.trust == 'static':
                __average_static(clients, extra_args.dataset)
            elif extra_args.trust == 'dynamic':
                __average_dynamic(clients)
            elif extra_args.trust == 'dynamic-thresh':
                __average_dynamic_threshold(clients)
            elif extra_args.trust == 'dynamic-top-k':
                __average_dynamic_top_k(clients, extra_args.k)
            elif 'ref' in extra_args.trust:
                res = torch.zeros((num_clients, num_clients))
                for model, _, _ in clients:
                    model.eval()
                for id1 in range(len(clients)):
                    for id2, (model, _, _) in enumerate(clients):
                        model.eval()
                        _, _, val_perplexity = eval(model, data['val'][id1], sequence_length, batch_size,
                                                    extra_args.device, max_num_batches=12, ctx=type_ctx)
                        res[id1, id2] = val_perplexity
                        model.train()
                for model, _, _ in clients:
                    model.train()
                res = -res
                if extra_args.trust == 'dynamic-ref':
                    __average_dynamic_ref(clients, res)
                elif extra_args.trust == 'dynamic-thresh-ref':
                    __average_dynamic_threshold_ref(clients, res)
                elif extra_args.trust == 'dynamic-top-k-ref':
                    __average_dynamic_top_k_ref(clients, res, extra_args.k)
            elif 'token' in extra_args.trust:
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

                res = torch.zeros((num_clients, num_clients))
                for id1 in range(len(clients)):
                    for id2 in range(len(clients)):
                        sim = 0
                        for j in range(4):
                            sim += torch.sum(torch.abs(logits[id1][j] - logits[id2][j])).item()
                        res[id1, id2] = sim / (4 * batch_size)

                res = F.normalize(res, p=1, dim=1)
                res = -res * 10
                if extra_args.trust == 'dynamic-token':
                    __average_dynamic_token(clients, res)
                elif extra_args.trust == 'dynamic-thresh-token':
                    __average_dynamic_threshold_token(clients, res)
                elif extra_args.trust == 'dynamic-top-k-token':
                    __average_dynamic_top_k_token(clients, res, extra_args.k)

        # from here it's only evaluation code, all the training is above
        t1 = time.time()
        dt = (t1 - t0) / num_clients
        for i in range(num_clients):
            model, opt, scheduler = clients[i]
            opt.zero_grad(set_to_none=True)

            if itr[i] % eval_freq == 0 or itr[i] == iterations:
                if distributed_backend.is_master_process():
                    epoch = substep[i] // num_substeps_per_epoch[i]

                    model.eval()
                    train_loss = loss.detach().cpu().item() * acc_steps
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                    val_acc, val_loss, val_perplexity = eval(model, data['val'][i], sequence_length, batch_size,
                                                             extra_args.device, max_num_batches=12, ctx=type_ctx)

                    print_string = f"{i}: {epoch}/{itr[i]} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                    print_string += f" [time per itr] {dt * 1000 / eval_freq:.2f}ms"
                    if scheduler is not None:
                        print_string += f" [lr] {current_lr:.5f}"
                    print(f'\r{print_string}')

                    stats['train_loss'][i].append(train_loss)
                    stats['val_loss'][i].append(val_loss)
                    stats['val_pp'][i].append(val_perplexity)
                    stats['val_acc'][i].append(val_acc)

                    if extra_args.wandb:
                        if i == (num_clients - 1):
                            wandb.log({
                                f"train/loss_mean": np.mean([stats['train_loss'][i][-1] for i in range(num_clients)]),
                                f"val/loss_mean": np.mean([stats['val_loss'][i][-1] for i in range(num_clients)]),
                                f"val/perplexity_mean": np.mean([stats['val_pp'][i][-1] for i in range(num_clients)]),
                                f"val/acc_mean": np.mean([stats['val_acc'][i][-1] for i in range(num_clients)]),
                            }, commit=False)
                        wandb.log({
                            f"iter_{i}": itr[i],
                            f"train/loss_{i}": train_loss,
                            f"val/loss_{i}": val_loss,
                            f"val/perplexity_{i}": val_perplexity,
                            f"val/acc_{i}": val_acc,
                            f"lr_{i}": current_lr,
                        }, commit=(i == (num_clients - 1)))

                    model.train()
        if itr[-1] % eval_freq == 0 or itr[-1] == iterations:
            for idx, c in enumerate(clients):
                model, _, _ = c
                torch.save(model.state_dict(), f'{idx}_{itr[-1]}')
        t0 = time.time()

    return stats
