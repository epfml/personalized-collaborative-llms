from argparse import Namespace
from typing import Union

from . import ddp
from . import single

BACKEND_TYPE_TO_MODULE_MAP = {
    "nccl": ddp.DataParallelDistributedBackend,
    None: single.SingleNodeBackend,
}


def make_backend_from_args(args: Namespace) -> Union[ddp.DataParallelDistributedBackend, single.SingleNodeBackend]:
    return BACKEND_TYPE_TO_MODULE_MAP[args.distributed_backend](args)


def registered_backends() -> dict.keys:
    return BACKEND_TYPE_TO_MODULE_MAP.keys()
