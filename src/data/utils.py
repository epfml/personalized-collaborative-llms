import os
from typing import Dict, List

import numpy as np

WIKI_PATH_EN = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.en")
WIKI_PATH_FR = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.fr")
WIKI_PATH_IT = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.it")
WIKI_PATH_DE = os.path.join(os.path.dirname(__file__), "wikipedia/20220301.de")

from .agnews import get_agnews_data
from .fed_cc_news import get_fed_cc_news
from .github_wiki import get_github_wikitext_data
from .wikitext_split import get_split_multi_data
from .three_multi import get_three_multi_data
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, List[np.ndarray] | np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing up to three keys: "train" and "val", and "ref", corresponding to the tokenized training,
     validation and reference data. """

    if args.dataset == "fed_cc_news":
        return get_fed_cc_news()

    elif args.dataset == "agnews_mixed":
        return get_agnews_data("mixed")
    elif args.dataset == "agnews_specific":
        return get_agnews_data("specific")
    elif args.dataset == "three_multi_specific":
        return get_three_multi_data("specific")
    elif args.dataset == "three_multi_mixed":
        return get_three_multi_data("mixed")
    elif args.dataset == "github_wiki_specific":
        return get_github_wikitext_data("specific")
    elif args.dataset == "github_wiki_mixed":
        return get_github_wikitext_data("mixed")

    elif args.dataset == "wikitext":
        return get_wikitext_data()
    elif args.dataset == "wiki_split_fr":
        return get_split_multi_data("fr")
    elif args.dataset == "wiki_split_it":
        return get_split_multi_data("it")
    elif args.dataset == "wiki_split_de":
        return get_split_multi_data("de")
    elif args.dataset == "wiki_split_en":
        return get_split_multi_data("en")
    else:
        raise NotImplementedError(f"Unknown dataset key {args.dataset}")
