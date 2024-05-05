from argparse import ArgumentParser

from data.utils import get_dataset

# Quick function to generate dataset in advance
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        choices=['wikitext', 'wiki_split_de', 'wiki_split_it', 'wiki_split_fr', 'wiki_split_en',
                                 'agnews_mixed', 'agnews_specific',
                                 'three_multi_specific', 'three_multi_mixed',
                                 'github_wiki_specific', 'github_wiki_mixed',
                                 'fed_cc_news'])
    _ = get_dataset(parser.parse_args())
