# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import re
import yaml
import glob
import random
import argparse
import importlib
from shutil import copyfile
from argparse import Namespace
#import torchaudio
from torch.distributed import is_initialized, get_world_size, get_rank
#-------------#
import torch
import numpy as np
#-------------#
from pretrain.runner import Runner
from utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override
os.environ["TORCH_HOME"] = "/home/u8786328/.torch"
#os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", "/home/u8786328/john/.torch/hub")

print(torch.hub.get_dir()) 

######################
# PRETRAIN ARGUMENTS #
######################
def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')
    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model')

    # upstream settings
    parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    parser.add_argument('-g', '--upstream_config', metavar='CONFIG_PATH', help='The yaml file for configuring the upstream model')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/pretrain
    parser.add_argument('-n', '--expname', help='Save experiment at expdir/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--multi_gpu', action='store_true', help='Enables multi-GPU training')

    args = parser.parse_args()

    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.expname}'

    if args.auto_resume:
        if os.path.isdir(args.expdir):
            ckpt_pths = glob.glob(f'{args.expdir}/states-*.ckpt')
            if len(ckpt_pths) > 0:
                args.past_exp = args.expdir

    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp

        print(f'[Runner] - Resume from {ckpt_pth}')

        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new, preserve_list=None):
            out_dict = vars(old)
            new_dict = vars(new)
            for key in list(new_dict.keys()):
                if key in preserve_list:
                    new_dict.pop(key)
            out_dict.update(new_dict)
            return Namespace(**out_dict)

        # overwrite args
        cannot_overwrite_args = [
            'mode', 'evaluate_split', 'override',
            'backend', 'local_rank', 'past_exp',
        ]
        args = update_args(args, ckpt['Args'], preserve_list=cannot_overwrite_args)
        os.makedirs(args.expdir, exist_ok=True)
        args.init_ckpt = ckpt_pth
        config = ckpt['Config']
        upstream_dirs = [u for u in os.listdir('pretrain/') if re.search(f'^{u}_|^{u}$', args.upstream)]
        config = f'pretrain/{upstream_dirs[0]}/config_runner.yaml'
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print('[Runner] - Start a new experiment')
        args.init_ckpt = None

        assert args.expname is not None
        if args.expdir is None:
            args.expdir = f'result/pretrain/{args.expname}'
        os.makedirs(args.expdir, exist_ok=True)

        upstream_dirs = [u for u in os.listdir('pretrain/') if re.search(f'^{u}_|^{u}$', args.upstream)]
        assert len(upstream_dirs) == 1

        if args.config is None:
            args.config = f'pretrain/{upstream_dirs[0]}/config_runner.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        if args.upstream_config is None:
            default_upstream_config = f'pretrain/{upstream_dirs[0]}/config_model.yaml'
            assert os.path.isfile(default_upstream_config)
            args.upstream_config = default_upstream_config
        if os.path.isfile(args.upstream_config):
            copyfile(args.upstream_config, f'{args.expdir}/config_model.yaml')
        else:
            raise FileNotFoundError('Wrong file path for model config.')

    if args.override is not None and args.override.lower() != "none":
        print(f"args {args}")
        print(f"config {config}")
        override(args.override, args, config)
        os.makedirs(args.expdir, exist_ok=True)
    
    if os.path.isfile(args.config):
        copyfile(args.config, f'{args.expdir}/config_runner.yaml')
    else:
        raise FileNotFoundError('Wrong file path for runner config.')

    return args, config


########
# MAIN #
########
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torchaudio.set_audio_backend('sox_io')
    hack_isinstance()

    # get config and arguments
    args, config = get_pretrain_args()

    ## When torch.distributed.launch is used
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(args.backend)

    if args.past_exp:
        # load checkpoint
        ckpt = torch.load(args.init_ckpt, map_location='cpu')

        now_use_ddp = is_initialized()
        original_use_ddp = ckpt['Args'].local_rank is not None
        assert now_use_ddp == original_use_ddp, f'{now_use_ddp} != {original_use_ddp}'
        
        if now_use_ddp:
            now_world = get_world_size()
            original_world = ckpt['WorldSize']
            assert now_world == original_world, f'{now_world} != {original_world}'



    # Save command
    if is_leader_process():
        with open(os.path.join(args.expdir, f'args_{get_time_tag()}.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

        with open(os.path.join(args.expdir, f'config_{get_time_tag()}.yaml'), 'w') as file:
            yaml.dump(config, file)


    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    runner = Runner(args, config)
    eval('runner.train')()
    runner.logger.close()


if __name__ == '__main__':
    main()
