import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from argparse import ArgumentParser
import torch
import argparse
from utils.training import train_il
from utils.conf import set_random_seed

def main():
    parser = ArgumentParser(description='UCLAD', allow_abbrev=False)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='seq-visa')#seq-mvtec,seq-visa
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument("--save_path", type=str, default='./results/visa/200_1/', help='path to save results')
    parser.add_argument('--t_c_arr', type=str, default=None)
    # parser.add_argument('--lambda_basis', type=float, default=0.01, help='weight for incremental basis regularization')

    args = parser.parse_args()

    args.device = torch.device(args.device)
    if args.seed is not None:
        set_random_seed(args.seed)

    if args.dataset == 'seq-mvtec':
        args.item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        # args.item_list = ['bottle', 'cable', 'capsule',]

    elif args.dataset == 'seq-visa':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        # args.item_list = ['pcb2','candle']

    #logger日志模块
    os.makedirs(args.save_path, exist_ok=True)
    txt_path = os.path.join(args.save_path, "train.log")
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    train_il(args)



if __name__ == '__main__':
    main()
