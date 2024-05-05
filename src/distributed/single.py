from argparse import Namespace
from contextlib import nullcontext
from typing import List

from torch import nn

from .backend import DistributedBackend


class SingleNodeBackend(DistributedBackend):

    def transform_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_context_for_microstep_forward(self, *args, **kwargs) -> nullcontext:
        return nullcontext()

    def get_adjusted_args_for_process(self, args: Namespace) -> Namespace:
        return args

    def is_master_process(self) -> bool:
        return True

    def get_raw_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_world_size(self) -> int:
        return 1

    def translate_model_parameter_name_for_node(self, parameter_name: str) -> List[str]:
        return [parameter_name]
