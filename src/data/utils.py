import numpy as np
from typing import Dict

from .agnews_mixed import get_agnews_mixed_data
from .agnews_specific import get_agnews_specific_data
from .fed_cc_news import get_fed_cc_news
from .github_wiki_mixed import get_github_wikitext_data_mixed
from .github_wiki_specific import get_github_wikitext_data_specific
from .split_wikitext import get_split_multi_data
from .three_multi_mixed import get_three_multi_data_mixed
from .three_multi_specific import get_three_multi_data_specific
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == 'agnews_mixed':
        return get_agnews_mixed_data()
    if args.dataset == 'agnews_specific':
        return get_agnews_specific_data()
    if args.dataset == 'three_multi_specific':
        return get_three_multi_data_specific()
    if args.dataset == 'three_multi_mixed':
        return get_three_multi_data_mixed()
    if args.dataset == 'github_wiki_specific':
        return get_github_wikitext_data_specific()
    if args.dataset == 'github_wiki_mixed':
        return get_github_wikitext_data_mixed()
    if args.dataset == 'split_wiki_fr':
        return get_split_multi_data('fr')
    if args.dataset == 'split_wiki_it':
        return get_split_multi_data('it')
    if args.dataset == 'split_wiki_de':
        return get_split_multi_data('de')
    if args.dataset == 'split_wiki_en':
        return get_split_multi_data('en')
    if args.dataset == 'fed_cc_news':
        return get_fed_cc_news()
    else:
        raise NotImplementedError(f"Unknown dataset key '{args.dataset}'")
