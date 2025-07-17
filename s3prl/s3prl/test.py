import os
import re
import yaml
import glob
import random
import argparse
import importlib
from shutil import copyfile
from argparse import Namespace
#-------------#
import torch
import numpy as np

# def get_pretrain_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')
#     parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    
#     args = parser.parse_args()
#     if args.past_exp:
 
#         print('=========================================================')
#         print(f'[Runner] - Resume from {args.past_exp}')

#         ckpt = torch.load(args.past_exp, map_location='cpu')

#     return args, 0

def main():
    # args, config = get_pretrain_args()
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')
    parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    
    args = parser.parse_args()
    if args.past_exp:
 
        print('=========================================================')
        print(f'[Runner] - Resume from {args.past_exp}')

        ckpt = torch.load(args.past_exp, map_location='cpu')

if __name__ == '__main__':
    main()


