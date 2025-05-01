import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
from datetime import timedelta
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.models_05 import *
from src.dataloader import TiggeMRMSPatchLoadDataset
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models_05 import GANs, gens, discs
from catalyst.data.sampler import DistributedSamplerWrapper

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import os

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"




import json
import argparse
import sys


def parseInputArgs():
    parser = argparse.ArgumentParser(
        description="specify_experiment_parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--experiment_config", type=str,
                        dest="config_path", help='Path to file containing configuration for the experiment')

    parser.add_argument("--ckpt_path", default=None, help="path to checkpoint to continue training from")
    return parser.parse_args()





args = {
        'seed': 1,
        'save_hparams':{'save_dir': '/LOCALDISK/smlzhuhai_jwma_1/savedmodels/',
                       'run_name': 'correctorgen-nc-LT12H-11channels-32patchsize-pad32-newloss-256batchsize-lossfss-weight-100-100-10/',
                       'run_number':0
                      },
        'gen_hparams': {'generator': 'correctorgen2',
                        'noise_shape' : (6, 1, 32, 32),
                        'zero_noise': True,
                        'input_channels' : 11,
                        'opt_hparams': {'gen_optimiser':'adam',
                                         'gen_lr': 5e-5, 'gen_freq' : 1,
                                        'b1':0.0, 'b2' : 0.9},
                        'gen_spectral_norm' : False,
                        'loss_hparams' : {'gen_loss':{"ens_mean_lr_L1_weighted":100, "ens_mean_L1_weighted": 100,"gen_hr_corrected_skill":10}} ,
                        'val_hparams' : {'val_nens':10}
                       },
        'train_hparams': {'epochs':51,
                          'gpus':[0,1,2,3],
                          'batch_size': 256
                         },
        'data_hparams': {
                        'train_dataset_path': "/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/train",
                        'valid_dataset_path': "/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/valid",
            'samples_vars':
                {'tp': 1,'lsp':1,'cp':1,'cape':1,'tcwv':1,'hm':1,'v':1,'u':1,'sp':1,'msl':1,'pad_tigge_channel': 1}
        }
}

json.dump(args, open('/LOCALDISK/smlzhuhai_jwma_1/CGAN/correctorgen-nc-LT12H-11channels-32patchsize-pad32-newloss-256batchsize-lossfss-weight-100-100-10.json', 'w'))




def train(input_args):
    # Load Experiment Args and Hyperparameters

    args = json.load(open(input_args.config_path))

    parser = argparse.ArgumentParser(args)
    parser.set_defaults(**args)
    args, _ = parser.parse_known_args()

    model_dir = args.save_hparams["save_dir"] + args.save_hparams["run_name"] + str(
        args.save_hparams["run_number"]) + "/"

    print("model_dir:", model_dir)
    sys.path.append(model_dir)
    print("sys path:", sys.path)


    print("Args loaded")
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.gen_hparams['generator'] = gens[args.gen_hparams['generator']]


    ## Load Data and set data params

    print("Loading data ... ")
    


    ds_train = TiggeMRMSPatchLoadDataset(args.data_hparams['train_dataset_path'], samples_vars=args.data_hparams['samples_vars'])
    ds_valid = TiggeMRMSPatchLoadDataset(args.data_hparams['valid_dataset_path'], samples_vars=args.data_hparams['samples_vars'])
    
    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.weights, len(ds_train))
    sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas = len(args.train_hparams['gpus']) if type(args.train_hparams['gpus'])==list else  args.train_hparams['gpus'], rank = 0)
    sampler_valid = torch.utils.data.SequentialSampler(ds_valid)
    sampler_valid = DistributedSamplerWrapper(sampler_valid, num_replicas = len(args.train_hparams['gpus']) if type(args.train_hparams['gpus'])==list else  args.train_hparams['gpus'], rank = 0)
    
    
    if type(args.train_hparams['gpus']) == list:
        batch_size = args.train_hparams['batch_size']//len(args.train_hparams['gpus'])
    else:
        batch_size = args.train_hparams['batch_size']//args.train_hparams['gpus']
    
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=32, pin_memory=True)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=32, pin_memory=True)
        
    print("len dl train", len(dl_train))
    print("Data loading complete")

    
    
    ddi_train = pickle.load(open("/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/traindataset_det_tp_pad_pw_cape_cp_msl_sp_v_u_gh_vb_ub_lsp.pkl", "rb"))
    args.gen_hparams['val_hparams']['ds_max'] = ddi_train.maxs.tp.values
    args.gen_hparams['val_hparams']['ds_min'] = ddi_train.mins.tp.values
    args.gen_hparams['val_hparams']['tp_log'] = ddi_train.tp_log
    print('this is the max min info:')
    print(args.gen_hparams['val_hparams']['ds_max'])
    print(args.gen_hparams['val_hparams']['ds_min'])
    del ddi_train


    
    ## Load Model
    if input_args.ckpt_path:
        state_dict=torch.load(input_args.ckpt_path)
        model = CorrectorGenerator(**args.gen_hparams)
        model.loss_hparams = args.gen_hparams['loss_hparams']
        model.opt_hparams = args.gen_hparams['opt_hparams']
        model.noise_shape = args.gen_hparams['noise_shape']
        print(model.opt_hparams['gen_lr'])
    else:
        model = CorrectorGenerator(**args.gen_hparams)
        save_model = torch.load('/LOCALDISK/smlzhuhai_jwma_1/savedmodels/correctornc-LT12H-11channels-32pachsize-pad32-newloss-075-lossmax20-tsfss0/epoch=49-step=10849.ckpt')
        model_dict = model.state_dict()
        state_dict = {f'gen{k[9:]}': save_model['state_dict'][k] for k in save_model['state_dict'] if
                      f'gen{k[9:]}' in model_dict.keys() and 'final' not in f'gen{k[9:]}'}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    ## Define trainer and logging

    save_dir = args.save_hparams['save_dir']

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir + args.save_hparams['run_name'] + str(args.save_hparams['run_number']) + '/',save_top_k=-1)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/LOCALDISK/smlzhuhai_jwma_1/logs/',
                                             name=args.save_hparams['run_name'],
                                             version=args.save_hparams['run_number'])

    if input_args.ckpt_path:
        from pytorch_lightning.plugins import DDPPlugin
        trainer = pl.Trainer(accelerator='ddp',
                             precision=16, gpus=args.train_hparams['gpus'],
                             max_epochs=args.train_hparams['epochs'],
                             callbacks=[checkpoint_callback],
                             plugins=DDPPlugin(find_unused_parameters=False),
                             replace_sampler_ddp=False,
                             #check_val_every_n_epoch=20,
                             logger=tb_logger,
                             resume_from_checkpoint=input_args.ckpt_path
                             )
    else:
        from pytorch_lightning.plugins import DDPPlugin
        trainer = pl.Trainer(accelerator='ddp',
                             precision=16, gpus=args.train_hparams['gpus'],
                             max_epochs=args.train_hparams['epochs'],
                             callbacks=[checkpoint_callback],
                             plugins=DDPPlugin(find_unused_parameters=False),
                             replace_sampler_ddp=False,
                             #check_val_every_n_epoch=20,
                             logger=tb_logger,
                             #                          auto_select_gpus=True
                             )

    print("Training model...")

    # Train
    trainer.fit(model, dl_train, dl_valid)

if __name__ == '__main__':
    input_args = parseInputArgs()
    input_args.config_path='/LOCALDISK/smlzhuhai_jwma_1/CGAN/correctorgen-nc-LT12H-11channels-32patchsize-pad32-newloss-256batchsize-lossfss-weight-100-100-10.json'
    #input_args.ckpt_path = '/LOCALDISK/smlzhuhai_jwma_1/savedmodels/correctorgen-nc-LT24H-11channels-32patchsize-pad32-newloss-256batchsize-lossmax20-weight-100-100/0/epoch=28-step=25142.ckpt'
    print(input_args.config_path)
    train(input_args)

