import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.models_05_loglrts import *   ####this is for the noise*0.2
from catalyst.data.sampler import DistributedSamplerWrapper

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pickle

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
    
import json
import argparse
import sys



def evaluate():
    # Load Experiment Args and Hyperparameters


    
    device = torch.device(f'cuda:1')
    print("device", device)
    model_dir = '/LOCALDISK/smlzhuhai_jwma_1/savedmodels/correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50-val/0/epoch=188-step=163673.ckpt' ### 配置modeldir 

    args_file = '/LOCALDISK/smlzhuhai_jwma_1/CGAN/correctorgan-11channels-LT12H-32patchsize-pad32-4nv-lr2e5-batchsize_256-newloss-loss-lrhr-ts-weight-100-100-50-50-val.json' ###final 配置脚本 changed here
    args = json.load(open(args_file))
    parser = argparse.ArgumentParser(args)
    parser.set_defaults(**args)
    args, _ = parser.parse_known_args()
    args.gan_hparams['generator'] = gens[args.gan_hparams['generator']]
    args.gan_hparams['discriminator'] = discs[args.gan_hparams['discriminator']]

    from src.dataloader import TiggeMRMSDataset
#     from run_src.utils import *
    from src.evaluation_o import par_gen_patch_eval_all

    #set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(args.seed)

    ## Load Data and set data params
    ds_test = pickle.load(open('/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_11channels_testdataset-stride32-dropnan-LT12H/traindataset_det_tp_pad_pw_cape_cp_msl_sp_v_u_gh_vb_ub_lsp.pkl', "rb"))  
    sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=128, sampler=sampler_test
    )

    print("Loading data ... ")

    gan = GANs[args.gan](**args.gan_hparams).load_from_checkpoint(model_dir)
    
    print("loaded gan")
    
    gen = gan.gen
    gen = gen.to(device)
    gen.train(False);

    print("Data loading complete")
#     print("ds test type", type(ds_test))
    ## Load Model

    metrics = par_gen_patch_eval_all(gen, dl_test, 10, ds_test.mins.tp.values, ds_test.maxs.tp.values, ds_test.tp_log, '188-epoch-LT12H-val-100-100-50-50', device)       
    print(metrics)

    pickle.dump(metrics, open('/LOCALDISK/smlzhuhai_jwma_1/evaluation/LT12H-188epoch-100-100-50-50-val-allscores.pkl', "wb"))

if __name__ == '__main__':
    evaluate()