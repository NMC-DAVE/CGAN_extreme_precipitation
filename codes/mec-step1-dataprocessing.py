import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models import *
from src.dataloader import *
from src.utils import *

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pickle
import json

data_dir = '/LOCALDISK/smlzhuhai_jwma_1/data/'
args = {'tigge_dir':'/LOCALDISK/smlzhuhai_jwma_1/data/forecast/ECMWFnc-8km-member-99E125E-17N43N/',
    'tigge_vars':['total_precipitation','largescalegrid_precipitation','convective_precipitation',
                  'convective_available_potential_energy','total_column_water',
                  'geopotential_height_500hpa','vwind_850hPa','uwind_850hPa',
                  'surface_pressure','mean_sea_level_pressure'],
    'mrms_dir':'/LOCALDISK/smlzhuhai_jwma_1/data/obs/QPE_03H_99E125E_17N43N/',
    'val_days':3,
    'lead_time':30,   ##changedhere
    'scale':True,
    'split':'train',
    'tp_log':0.01,
    'patch_size': 256, ##8*32 patch_size
    'idx_stride': 16, ##stride 1/2
    'ensemble_mode':'stack_by_variable',
    'pad_tigge': 32, ##pad 后512km
    'pad_tigge_channel':True
     }

save_dir = '/LOCALDISK/smlzhuhai_jwma_1/traindata/pachsize_32_11channels-patchsize256-pad32-leadtime-30H/' ###changgedhere
dataset_name = 'det_tp_pad_pw_cape_cp_msl_sp_v_u_gh_vb_ub_lsp'
ds_train = TiggeMRMSDataset(**args)
save_images(ds_train, save_dir, 'train',0)
pickle.dump(ds_train, open(save_dir + f"traindataset_{dataset_name}.pkl", "wb"))
pickle.dump(args, open(save_dir + f"traindataset_{dataset_name}_args.pkl", "wb"))

val_args = args
val_args['maxs'] = ds_train.maxs
val_args['mins'] = ds_train.mins
val_args['split'] = 'valid'
ds_valid = TiggeMRMSDataset(**val_args)
save_images(ds_valid, save_dir, 'valid',0)
pickle.dump(ds_valid, open(save_dir + f"validdataset_{dataset_name}.pkl", "wb"))
pickle.dump(val_args, open(save_dir + f"validdataset_{dataset_name}_args.pkl", "wb"))
val_args = pickle.load(open(save_dir + f"validdataset_{dataset_name}_args.pkl", 'rb'))
print(val_args)
