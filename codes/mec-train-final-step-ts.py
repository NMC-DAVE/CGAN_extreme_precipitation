import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.models_05_loglrts_val import *
from src.dataloader import TiggeMRMSPatchLoadDataset
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader




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





args = {'gan': 'corrector',
        'seed': 1,
        'save_hparams':{'save_dir': '/LOCALDISK/smlzhuhai_jwma_1/savedmodels/',
                       'run_name': 'correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50-val/',  ###changedhere
                       'run_number':2
                      },
        'gan_hparams': {'generator': 'correctorgen2',
                        'discriminator': 'leindisc',
                        'noise_shape' : (6, 1, 32, 32),
                        'zero_noise': False,
                        'input_channels' : 11,
                        'opt_hparams': {'gen_optimiser':'adam', 'disc_optimiser':'adam',
                                        'disc_lr' : 2e-5, 'gen_lr': 2e-5, 'gen_freq' : 1,
                                        'disc_freq':5, 'b1':0.0, 'b2' : 0.9},
                        'disc_spectral_norm' : False,
                        'gen_spectral_norm' : False,
                        'loss_hparams' : {'gen_loss':{"wasserstein":1,"ens_mean_lr_L1_weighted":100, "ens_mean_L1_weighted": 100,"gen_hr_corrected_ts":50,"gen_lr_corrected_ts":50}, 'disc_loss':{'wasserstein':1,'gradient_penalty': 10}} ,
                        'val_hparams' : {'val_nens':10}
                       },
        'train_hparams': {'epochs':201,
                          'gpus':[0,1,2,3],
                          'batch_size': 256
                         },
        'data_hparams': {
                        'train_dataset_path': "/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/train", ###changedhere
                        'valid_dataset_path': "/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/valid", ###changedhere
            'samples_vars':
                {'tp': 1,'lsp':1,'cp':1,'cape':1,'tcwv':1,'hm':1,'v':1,'u':1,'sp':1,'msl':1,'pad_tigge_channel': 1}
        }
}

json.dump(args, open('/LOCALDISK/smlzhuhai_jwma_1/CGAN/correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50-val.json', 'w')) ###changedhere



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


    from src.dataloader import TiggeMRMSDataset
    #     from run_src.utils import *

    print("Args loaded")
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    



 



    args.gan_hparams['generator'] = gens[args.gan_hparams['generator']]
    args.gan_hparams['discriminator'] = discs[args.gan_hparams['discriminator']]


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
    
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=16, pin_memory=True)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=16, pin_memory=True)
        
    print("len dl train", len(dl_train))
    print("Data loading complete")

    
    
    ddi_train = pickle.load(open("/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_10channels-new-pad32/traindataset_det_tp_pad_pw_cape_cp_msl_sp_v_u_gh_vb_ub_lsp.pkl", "rb")) ###changedhere
    args.gan_hparams['val_hparams']['ds_max'] = ddi_train.maxs.tp.values
    args.gan_hparams['val_hparams']['ds_min'] = ddi_train.mins.tp.values
    args.gan_hparams['val_hparams']['tp_log'] = ddi_train.tp_log
    print('this is the max min info:')
    print(args.gan_hparams['val_hparams']['ds_max'])
    print(args.gan_hparams['val_hparams']['ds_min'])
    print(args.gan_hparams['val_hparams']['tp_log'])
    del ddi_train
    print('this is the gans info:')
    print(args.gan_hparams)

    print("Data loading complete")
    ## Load Model
    if input_args.ckpt_path:
        model = GANs[args.gan].load_from_checkpoint(input_args.ckpt_path)
        model.loss_hparams = args.gan_hparams['loss_hparams']
        model.opt_hparams = args.gan_hparams['opt_hparams']
        model.noise_shape = args.gan_hparams['noise_shape']
        print(model.opt_hparams['disc_lr'])
        print(model.opt_hparams['gen_lr'])
    else:
        model = GANs[args.gan](**args.gan_hparams)
        save_model = torch.load('/LOCALDISK/smlzhuhai_jwma_1/savedmodels/correctorgen-nc-LT12H-11channels-32patchsize-pad32-newloss-256batchsize-loss-lhr-ts-weight-100-100-20-20/0/epoch=49-step=43299.ckpt') ###changedhere
        model_dict = model.state_dict()
        state_dict = {k: save_model['state_dict'][k] for k in save_model['state_dict'] if
                  k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    ## Define trainer and logging

    save_dir = args.save_hparams['save_dir']

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir + args.save_hparams['run_name'] + str(args.save_hparams['run_number']) + '/',
        save_top_k=-1)


    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/LOCALDISK/smlzhuhai_jwma_1/logs/',
                                             name=args.save_hparams['run_name'],
                                             version=args.save_hparams['run_number'])

    if input_args.ckpt_path:
        trainer = pl.Trainer(accelerator='ddp',
                             precision=16, gpus=args.train_hparams['gpus'],
                             max_epochs=args.train_hparams['epochs'],
                             callbacks=[checkpoint_callback],
                             replace_sampler_ddp=False,
                             #check_val_every_n_epoch=20,
                             logger=tb_logger,
                             resume_from_checkpoint=input_args.ckpt_path
                             )
    else:
        trainer = pl.Trainer(accelerator='ddp',
                             precision=16, gpus=args.train_hparams['gpus'],
                             max_epochs=args.train_hparams['epochs'],
                             callbacks=[checkpoint_callback],
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
    input_args.config_path='/LOCALDISK/smlzhuhai_jwma_1/CGAN/correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50-val.json' ###changedhere
    #input_args.ckpt_path='/LOCALDISK/smlzhuhai_jwma_1/savedmodels/correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50/0/epoch=110-step=96125.ckpt'
    print(input_args.config_path)
    train(input_args)

