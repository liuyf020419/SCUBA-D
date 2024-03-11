import sys,os
import argparse
from tqdm import tqdm
import contextlib
import logging
import time
from typing import Any, Dict, List
from collections import OrderedDict
from ml_collections import ConfigDict
import json

import numpy as np

import torch
import torch.nn as nn

from protdiff.models.priorddpm import PriorDDPM
from protdiff.dataset import ProtDiffParDataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)


noising_mode_dict = {
        'refine_prior': 0, 
        'gen_from_noise': 2,
        'loopsampling': 2,
        'structure_gen_from_sstype': 4
        }


model_name = 'sketch_mask90_domain_tune_noceaatype_minbatch_sketch_mask'

def build_parser():
    parser = argparse.ArgumentParser(description='SCUBA_Diff')
    parser.add_argument('--gen_dir', type=str, default=None, help='generate dir')
    parser.add_argument('--model_path', type=str, default=f'savedir/{model_name}/checkpoint/checkpoint_clean.pt', help='path to checkpoint file')
    parser.add_argument('--fixed_model_path', type=str, default=None, help='path to fixed checkpoint file')
    parser.add_argument('--root_dir', type=str, default=f'savedir/{model_name}', help='project path')
    parser.add_argument('--test_list', type=str, default=None, help='test list')
    parser.add_argument('--gen_tag', type=str, default='', help='gen_tag')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    parser.add_argument('--sample_from_raw_pdbfile', action='store_true', help='sample from raw pdbfile or processed file')
    parser.add_argument('--max_sample_num', type=int, default=10000000, help='maximum number of samples for testing or application')
    
    parser.add_argument('--noising_mode', type=str, default=None, help='noising mode for testing model')
    parser.add_argument('--rmsd_cutoff', type=str, default=None, help='rmsd_cutoff for testing model (FROMAT: MINRMSD+MAXRMSD)')
    parser.add_argument('--write_pdbfile', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--epoch_num', type=int, default=3, help='epoch number for sampling')
    parser.add_argument('--step_size', type=int, default=3, help='iteration number per epoch')
    parser.add_argument('--pdb_root', type=str, default=None, help='pdb root for application')
    parser.add_argument('--return_traj', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--update_init_traj', action='store_true', help='update initial traj per epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize for each protein')
    parser.add_argument('--diff_noising_scale', type=float, default=1.0, help='noising scale for diffusion')
    parser.add_argument('--iterate_mode', type=str, default='structure_gen_from_sstype', help='iterate mode')

    return parser


def load_config(path)->ConfigDict:
    return ConfigDict(json.loads(open(path).read()))


def load_checkpoint(checkpoint_path, model):
    last_cp= checkpoint_path
    if not os.path.exists(last_cp):
        logger.error(f'checkpoint file {last_cp} not exist, ignore load_checkpoint')
        return
    with open(last_cp,'rb') as f:
        logger.info(f'load checkpoint: {checkpoint_path}')
        state = torch.load(f, map_location=torch.device("cpu"))
    model.load_state_dict(state['model'])
    return model


def expand_batch(batch: dict, expand_size: int):
    new_batch = {}
    for k in batch.keys():
        if isinstance(batch[k], list):
            new_batch[k] = batch[k] * expand_size
        elif (('traj' in k) or ('pos_center' in k)):
            new_batch[k] = batch[k][0]
        else:
            shape_len = len(batch[k].shape)-1
            repeat_shape = [expand_size] + [1] * shape_len
            new_batch[k] = batch[k].repeat(*repeat_shape)

    return new_batch

    
def main(args):
    config_file = os.path.join(args.root_dir, 'config.json')
    assert os.path.exists(config_file), f'config file not exist: {config_file}'
    config= load_config(config_file)

    # modify config for inference
    config.data.train_mode = False
    config.args = args

    logger.info('start preprocessing...')
    model_config = config.model
    global_config = config.model.global_config
    data_config = config.data

    model = PriorDDPM(model_config, global_config, data_config)
    
    if args.model_path is not None:
        last_cp = args.model_path
    else:
        checkpoint_dir = f'{args.root_dir}/checkpoint'
        last_cp= os.path.join(checkpoint_dir, f"checkpoint_last.pt")
    logger.info(f'logging model checkpoint')
    _ = load_checkpoint(last_cp, model)

    model.eval()
    model.cuda()

    dataset = ProtDiffParDataset(config, args.test_list, batch_size=args.batch_size)
    count = 0

    if (len(dataset) < args.max_sample_num):
        args.max_sample_num = len(dataset)

    for data in tqdm(dataset):
        if count >= args.max_sample_num:
            break
        count+=1
        name = pdbname = data['pdbname']
        pdblen = data['len'].item()
        batch = {}

        for k, v in data.items():
            if k not in ['len', 'loss_mask', 'pdbname', 'noising_mode_idx', 'cath_architecture', \
                'pdb_raw_idx', 'fix_condition', 'noising_mode', 'output_prefix', 'cfg_file']:
                batch[k] = v[None].cuda(non_blocking=True)
            elif k in ['pdbname', 'noising_mode_idx', 'pdb_raw_idx', 'fix_condition', \
                'noising_mode', 'output_prefix', 'cfg_file']:
                batch[k] = [v]
            else:
                batch[k] = v.cuda(non_blocking=True)
        
        output_prefix = batch['output_prefix'][0]
        output_dir = f'{output_prefix}/{os.path.basename(output_prefix)}'
        pdbname = batch['pdbname'][0]
        cfg_file = batch['cfg_file'][0]

        logger.info(f'pdb name: {pdbname}; length: {pdblen}')
        os.makedirs(f'{output_prefix}', exist_ok=True)
        os.system(f'cp {cfg_file} {output_prefix}')

        with torch.no_grad():
            noising_mode_idx = batch['noising_mode'][0]
            iterate_mode_idx = noising_mode_dict[args.iterate_mode]
            fix_condition = torch.tensor(batch['fix_condition'][0])

            batch = expand_batch(batch, args.batch_size)
            model.sampling(
                batch, output_dir, args.step_size, 
                noising_mode_idx, fix_condition, 
                return_traj=args.return_traj, 
                epoch=args.epoch_num,
                diff_noising_scale=args.diff_noising_scale, 
                iterate_mode=iterate_mode_idx)


if __name__=='__main__':
    parser= build_parser()
    args= parser.parse_args()
    main(args)

