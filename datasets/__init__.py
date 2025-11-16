from datasets.seq_mvtec import SequentialMVTec
from datasets.seq_visa import SequentialVisA
from datasets.utils.continual_dataset import ContinualDataset
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from argparse import Namespace

NAMES = {
    SequentialMVTec.NAME: SequentialMVTec,
    SequentialVisA.NAME: SequentialVisA,
}

def get_dataset(args: Namespace) -> ContinualDataset:

    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
