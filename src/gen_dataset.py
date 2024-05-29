from argparse import Namespace

from data.utils import get_dataset
import sys

# Quick function to generate dataset in advance
if __name__ == '__main__':
    args = Namespace()
    args.dataset = sys.argv[1]
    args.train_size = int(sys.argv[2])
    _ = get_dataset(args)
