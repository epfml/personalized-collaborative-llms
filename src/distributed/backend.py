from argparse import Namespace
from typing import List, Any

from torch import nn


class DistributedBackend(object):

    def __init__(self, args: Namespace) -> None:
        pass

    def transform_model(self, model: nn.Module) -> Any:
        raise NotImplementedError

    def get_context_for_microstep_forward(self, model: nn.Module, microstep_idx: int, gradient_accumulation_steps: int):
        raise NotImplementedError

    def is_master_process(self) -> bool:
        raise NotImplementedError

    def get_adjusted_args_for_process(self, args: Namespace) -> Namespace:
        raise NotImplementedError

    def get_raw_model(self, model: Any) -> nn.Module:
        raise NotImplementedError

    def translate_model_parameter_name_for_node(self, parameter_name: str) -> List[str]:
        raise NotImplementedError

    def get_world_size(self) -> int:
        raise NotImplementedError

    def finalize(self):
        pass
