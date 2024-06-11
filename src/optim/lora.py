import json
import time
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from torch import Tensor
import torch.nn.functional as F

from .utils import eval, get_batch


def train_lora(clients, data, iterations, acc_steps, batch_size, sequence_length, eval_freq,
               distributed_backend, extra_args):
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

        # distribute gradient
        if itr[-1] % extra_args.trust_freq == 0 and itr[-1] >= extra_args.pretraining_rounds - 1:
            if extra_args.trust == 'local':
                pass
            elif extra_args.trust == 'fed-avg':
                __average(clients)
            elif extra_args.trust == 'fed-avg-ft':
                __average(clients)
            elif extra_args.trust == 'val_sim':
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
                __average_validation_set(clients, res)
            elif extra_args.trust == 'oracle':
                __average_oracle(clients, torch.tensor(data['samples_size']))
            elif extra_args.trust == 'oracle-1':
                __average_oracle(clients, torch.ones(len(clients)))
            else:
                raise NotImplementedError(f"Trust method {extra_args.trust} not implemented")

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
        t0 = time.time()

    return stats


def __weighted_average(clients, trust_weights) -> None:
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


def __average(clients) -> None:
    trust_weights = torch.zeros((len(clients), len(clients)))
    trust_weights = torch.fill(trust_weights, 1 / len(clients))
    __weighted_average(clients, trust_weights)


def __model_dot_product(client1, client2):
    score = 0
    total_params = 0
    for (name1, param1), (name2, param2) in zip(client1.named_parameters(), client2.named_parameters()):
        if name1 != name2:
            raise NameError(f'Should be the: {name1} != {name2}')
        if param1.requires_grad:
            sim = F.cosine_similarity(param1, param2)
            total_params += sim.size(0)
            score += torch.sum(sim).detach().item()

    return score / total_params


def __clients_similarity(clients) -> Tensor:
    trust_weight = torch.zeros((len(clients), len(clients)))
    for idx1, (model1, _, _) in enumerate(clients):
        for idx2, (model2, _, _) in enumerate(clients):
            if idx2 <= idx1:
                score = __model_dot_product(model1, model2)
                trust_weight[idx1, idx2] = score
                trust_weight[idx2, idx1] = score
    return trust_weight

def __average_oracle(clients, samples_size) -> None:
    trust_weights = torch.zeros((len(clients), len(clients)))
    for idx1 in range(len(clients)):
        for idx2 in range(len(clients)):
            if (idx1 % 3) == (idx2 % 3):
                trust_weights[idx1, idx2] = 1.
            else:
                trust_weights[idx1, idx2] = 0.

    trust_weights /= (len(clients) // 3)
    print(f"Trust_weights: {trust_weights}")
    trust_weights *= samples_size
    trust_weights /= trust_weights.sum(dim=1)
    print(f"Row stochastic trust_weights, scaled by samples size: {trust_weights}")
    __weighted_average(clients, trust_weights)

def __average_oracle_noised(clients, samples_size) -> None:
    trust_weights = torch.zeros((len(clients), len(clients)))
    for idx1 in range(len(clients)):
        for idx2 in range(len(clients)):
            if (idx1 % 3) == (idx2 % 3):
                trust_weights[idx1, idx2] = 1.
            else:
                trust_weights[idx1, idx2] = 0.

    trust_weights /= (len(clients) // 3)
    print(f"Trust_weights: {trust_weights}")
    trust_weights *= samples_size
    trust_weights /= trust_weights.sum(dim=1)
    print(f"Row stochastic trust_weights, scaled by samples size: {trust_weights}")
    __weighted_average(clients, trust_weights)


def __average_validation_set(clients, trust_weights) -> None:
    trust_weights = F.softmax(trust_weights, dim=1)
    __weighted_average(clients, trust_weights)