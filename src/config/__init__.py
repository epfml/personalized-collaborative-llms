from argparse import Namespace, ArgumentParser
from typing import List

from . import lora

CONFIG_FORMAT_TO_MODULE_MAP = {
    'lora': lora,
}


def parse_args_with_format(format: str, base_parser: ArgumentParser, args: List[str],
                           namespace: Namespace) -> Namespace:
    return CONFIG_FORMAT_TO_MODULE_MAP[format].parse_args(base_parser, args, namespace)


def registered_formats() -> dict.keys:
    return CONFIG_FORMAT_TO_MODULE_MAP.keys()
