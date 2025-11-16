import random
import torch
import numpy as np

def base_path() -> str:
    return './data/'

def base_mvtec_path() -> str:
    return '/media/LiuYuyao/Dataset/mvtec/'

def base_visa_path() -> str:
    return '/media/LiuYuyao/Dataset/visa/'

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False