from typing import Dict

import numpy as np

from .agnews_specific import get_agnews_specific_data
from .three_multi_specific_1 import get_three_multi_data_specific_1
from .three_multi_specific_2 import get_three_multi_data_specific_2
from .three_multi_specific_3 import get_three_multi_data_specific_3
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == 'agnews_specific':
        return get_agnews_specific_data()
    if args.dataset == 'three_multi_specific_1':
        return get_three_multi_data_specific_1()
    if args.dataset == 'three_multi_specific_2':
        return get_three_multi_data_specific_2()
    if args.dataset == 'three_multi_specific_3':
        return get_three_multi_data_specific_3()
    else:
        raise NotImplementedError(f"Unknown dataset key '{args.dataset}'")
