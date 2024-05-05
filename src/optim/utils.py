from contextlib import nullcontext
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast


def get_batch(data: np.ndarray, seq_length: int, batch_size: int, device: str = 'cpu') -> Tuple[Tensor, Tensor]:
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def eval(model: nn.Module, data_tensor: np.ndarray, sequence_length: int, batch_size: int, device: str = 'cpu',
         max_num_batches: int = 24, ctx: Union[nullcontext, autocast] = nullcontext()) -> Tuple[float, float, float]:
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity
