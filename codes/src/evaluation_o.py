import xarray as xr
from sklearn import metrics
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd
import xskillscore as xs
from dask.diagnostics import ProgressBar
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import date
import subprocess
import torch
import torch.nn as nn
from src.dataloader import log_retrans
import tqdm.notebook as tqdm
from pytictoc import TicToc
from multiprocess import Pool
import multiprocessing as mp
from tqdm import tqdm
from src.regrid import regrid
import torch.nn.functional as F

"""
Eval Functions
"""
##评分计算
def prep_clf(obs, pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives




###ecfss verification
def ecfss_interp_patch_eval(dl_test, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    

  

    preds_fss_10_window4 = []
    preds_fss_20_window4 = []
    preds_fss_30_window4 = []

    preds_fss_10_window10 = []
    preds_fss_20_window10 = []
    preds_fss_30_window10 = []

    preds_fss_10_window20 = []
    preds_fss_20_window20 = []
    preds_fss_30_window20 = []
    
  
    
    
    
    print(f"Total batches: {len(dl_test)}")
    


            
    for batch_idx, (x,y) in enumerate(dl_test):
        x = x.to(device)
        preds = F.interpolate(x, size = (256, 256), mode='bilinear').detach().to('cpu').numpy().squeeze()
        predsm = preds
        predsa= np.array([preds, predsm])


        truth = y.numpy().squeeze(1)
    
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        prednew = xr.DataArray(
                predsa,
                dims=['member','sample', 'lat', 'lon'],
                name='tp'
            )
        
        
        #print(prednew)
        truth = truth * (ds_max - ds_min) + ds_min

        prednew = prednew * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            prednew = log_retrans(prednew, tp_log)



        mean_fss_10_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=10, device=device)
        preds_fss_10_window10.append(mean_fss_10_window10)
        print(preds_fss_10_window10)
        
        mean_fss_20_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=10, device=device)
        preds_fss_20_window10.append(mean_fss_20_window10)
        print(preds_fss_20_window10)

        mean_fss_30_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=10, device=device)
        preds_fss_30_window10.append(mean_fss_30_window10)
        print(preds_fss_30_window10)



        mean_fss_10_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=4, device=device)
        preds_fss_10_window4.append(mean_fss_10_window4)
        print(preds_fss_10_window4)
        
        mean_fss_20_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=4, device=device)
        preds_fss_20_window4.append(mean_fss_20_window4)
        print(preds_fss_20_window4)

        mean_fss_30_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=4, device=device)
        preds_fss_30_window4.append(mean_fss_30_window4)
        print(preds_fss_30_window4)

        mean_fss_10_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=20, device=device)
        preds_fss_10_window20.append(mean_fss_10_window20)
        print(preds_fss_10_window20)
        
        mean_fss_20_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=20, device=device)
        preds_fss_20_window20.append(mean_fss_20_window20)
        print(preds_fss_20_window20)

        mean_fss_30_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=20, device=device)
        preds_fss_30_window20.append(mean_fss_30_window20)
        print(preds_fss_30_window20)
      
    
    metrics = {
               "fss_10_window10": np.nanmean(preds_fss_10_window10),
               "fss_20_window10": np.nanmean(preds_fss_20_window10),
               "fss_30_window10": np.nanmean(preds_fss_30_window10),

               "fss_10_window4": np.nanmean(preds_fss_10_window4),
               "fss_20_window4": np.nanmean(preds_fss_20_window4),
               "fss_30_window4": np.nanmean(preds_fss_30_window4),

               "fss_10_window20": np.nanmean(preds_fss_10_window20),
               "fss_20_window20": np.nanmean(preds_fss_20_window20),
               "fss_30_window20": np.nanmean(preds_fss_30_window20),
               
              }
    
    
    return metrics

###end ecmwf verfication

#####compute every epoch scores to choose the epoch
def compute_metrics_everyepoch(truth, preds, sample):    ##CGAN评分计算
 
    sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
    hits_10, misses_10, falsealarms_10, correctnegatives_10=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),10)
    hits_20, misses_20, falsealarms_20, correctnegatives_20=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),20)
    hits_30, misses_30, falsealarms_30, correctnegatives_30=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),30)
    
    
    return ( sample_rmse, 
             hits_10, misses_10, falsealarms_10, correctnegatives_10,
             hits_20, misses_20, falsealarms_20, correctnegatives_20,
             hits_30, misses_30, falsealarms_30, correctnegatives_30)




def par_gen_patch_everyepoch(gen, dl_test, nens, ds_min, ds_max, tp_log, device):    ##cgan评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    rmse = []
    pred_means = []
    truth_means = []
    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []
    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []
    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            rmse.append(res[0])
            hits_10.append(res[1])
            misses_10.append(res[2])
            falsealarms_10.append(res[3])
            correctnegatives_10.append(res[4])
            hits_20.append(res[5])
            misses_20.append(res[6])
            falsealarms_20.append(res[7])
            correctnegatives_20.append(res[8])
            hits_30.append(res[9])
            misses_30.append(res[10])
            falsealarms_30.append(res[11])
            correctnegatives_30.append(res[12])


            
        print("batch complete")
        print(f"current len of rmse {len(rmse)}")
            
    for batch_idx, (x,y,href) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)


        pool.starmap_async(compute_metrics_everyepoch, [(truth, preds, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    
    metrics = {
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
                "CSI_10":np.sum(hits_10) / (np.sum(hits_10) + np.sum(falsealarms_10) + np.sum(misses_10)),
                "CSI_20":np.sum(hits_20) / (np.sum(hits_20) + np.sum(falsealarms_20) + np.sum(misses_20)),
                "CSI_30":np.sum(hits_30) / (np.sum(hits_30) + np.sum(falsealarms_30) + np.sum(misses_30)),
              }
    
    return metrics

####end every epoch





def compute_metrics_junxu(truth, preds, truth_pert, preds_pert, sample):    ##CGAN评分计算
    sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
    sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
    rhist = xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values
    
    rel20 = xs.reliability(truth.sel(sample=sample)>20,(preds.sel(sample=sample)>20).mean('member'))
    rel20 = xr.where(np.isnan(rel20), 0, rel20)
    rel20['relative_freq'] = rel20
    
    rel30 = xs.reliability(truth.sel(sample=sample)>30,(preds.sel(sample=sample)>30).mean('member'))
    rel30 = xr.where(np.isnan(rel30), 0, rel30)
    rel30['relative_freq'] = rel30

    
    sample_brier_10 = xs.brier_score(truth.sel(sample=sample) > 10.0, (preds.sel(sample=sample) > 10.0).mean('member'), dim=['lat', 'lon'])
    
    
    sample_brier_20 = xs.brier_score(truth.sel(sample=sample) > 20.0, (preds.sel(sample=sample) > 20.0).mean('member'), dim=['lat', 'lon'])
        
    sample_brier_30 = xs.brier_score(truth.sel(sample=sample) > 30.0, (preds.sel(sample=sample) > 30.0).mean('member'), dim=['lat', 'lon'])

    hits_10, misses_10, falsealarms_10, correctnegatives_10=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),10)
    hits_20, misses_20, falsealarms_20, correctnegatives_20=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),20)
    hits_30, misses_30, falsealarms_30, correctnegatives_30=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),30)
    
    
    return (sample_crps,  sample_rmse, rhist, rel20, rel30, sample_brier_10, sample_brier_20, sample_brier_30,
             hits_10, misses_10, falsealarms_10, correctnegatives_10,
             hits_20, misses_20, falsealarms_20, correctnegatives_20,
             hits_30, misses_30, falsealarms_30, correctnegatives_30)

###grapes verification
def compute_metrics_grapes(truth, preds, sample):    ##grapes评分计算
    sample_rmse = xs.rmse(preds.sel(sample=sample), truth.sel(sample=sample), dim=['lat', 'lon']).values

    hits_1, misses_1, falsealarms_1, correctnegatives_1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),1)
    hits_5, misses_5, falsealarms_5, correctnegatives_5=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),5)
    hits_10, misses_10, falsealarms_10, correctnegatives_10=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),10)    
    hits_20, misses_20, falsealarms_20, correctnegatives_20=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),20)
    hits_30, misses_30, falsealarms_30, correctnegatives_30=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),30)
    hits_50, misses_50, falsealarms_50, correctnegatives_50=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),50)
    
    
    return (sample_rmse, 
            hits_1, misses_1, falsealarms_1, correctnegatives_1,
            hits_5, misses_5, falsealarms_5, correctnegatives_5,
            hits_10, misses_10, falsealarms_10, correctnegatives_10,
             hits_20, misses_20, falsealarms_20, correctnegatives_20,
             hits_30, misses_30, falsealarms_30, correctnegatives_30,
             hits_50, misses_50, falsealarms_50, correctnegatives_50)



def hreffss_patch_eval_grapes(dl_test, ds_min, ds_max, tp_log, device): ####grapes评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    

    preds_fss_10_window4 = []
    preds_fss_20_window4 = []
    preds_fss_30_window4 = []

    preds_fss_10_window10 = []
    preds_fss_20_window10 = []
    preds_fss_30_window10 = []

    preds_fss_10_window20 = []
    preds_fss_20_window20 = []
    preds_fss_30_window20 = []

    
            
    for batch_idx, (x,y) in enumerate(dl_test):

#         preds = x.squeeze()
#         print(preds.shape)
        preds = x.numpy()
        truth = y.numpy().squeeze(1)
        predsm = preds
        predsa= np.array([preds, predsm])
        
#        print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )

        
        prednew = xr.DataArray(
                predsa,
                dims=['member','sample', 'lat', 'lon'],
                name='tp'
            )
        

        truth = truth * (ds_max - ds_min) + ds_min


        if tp_log:
            truth = log_retrans(truth, tp_log)
        
        mean_fss_10_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=20, device=device)
        preds_fss_10_window20.append(mean_fss_10_window20)
        mean_fss_20_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=20, device=device)
        preds_fss_20_window20.append(mean_fss_20_window20)
        mean_fss_30_window20 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=20, device=device)
        preds_fss_30_window20.append(mean_fss_30_window20)

        mean_fss_10_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=10, device=device)
        preds_fss_10_window10.append(mean_fss_10_window10)
        mean_fss_20_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=10, device=device)
        preds_fss_20_window10.append(mean_fss_20_window10)
        mean_fss_30_window10 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=10, device=device)
        preds_fss_30_window10.append(mean_fss_30_window10)

        mean_fss_10_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=4, device=device)
        preds_fss_10_window4.append(mean_fss_10_window4)
        mean_fss_20_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=4, device=device)
        preds_fss_20_window4.append(mean_fss_20_window4)
        mean_fss_30_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=4, device=device)
        preds_fss_30_window4.append(mean_fss_30_window4)

        
        
    
    metrics = {

               "fss_10_window10": np.nanmean(preds_fss_10_window10),
               "fss_20_window10": np.nanmean(preds_fss_20_window10),
               "fss_30_window10": np.nanmean(preds_fss_30_window10),

               "fss_10_window4": np.nanmean(preds_fss_10_window4),
               "fss_20_window4": np.nanmean(preds_fss_20_window4),
               "fss_30_window4": np.nanmean(preds_fss_30_window4),

               "fss_10_window20": np.nanmean(preds_fss_10_window20),
               "fss_20_window20": np.nanmean(preds_fss_20_window20),
               "fss_30_window20": np.nanmean(preds_fss_30_window20),

                
              }
    
    
    return metrics

def href_patch_eval_grapes(dl_test, ds_min, ds_max, tp_log, device): ####grapes评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    t = TicToc()
    rmse = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []

    preds_fss_5_window4 = []
    preds_fss_10_window4 = []
    preds_fss_20_window4 = []
    preds_fss_30_window4 = []
    preds_fss_50_window4 = []

    preds_fss_5_window8 = []
    preds_fss_10_window8 = []
    preds_fss_20_window8 = []
    preds_fss_30_window8 = []
    preds_fss_50_window8 = []
    
    preds_fss_5_window16 = []
    preds_fss_10_window16 = []
    preds_fss_20_window16 = []
    preds_fss_30_window16 = []
    preds_fss_50_window16 = []

    preds_fss_5_window32 = []
    preds_fss_10_window32 = []
    preds_fss_20_window32 = []
    preds_fss_30_window32 = []
    preds_fss_50_window32 = []

    hits_1 = []
    misses_1 = []
    falsealarms_1 = []
    correctnegatives_1 = []

    hits_5 = []
    misses_5 = []
    falsealarms_5 = []
    correctnegatives_5 = []

    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []

    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []

    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []

    hits_50 = []
    misses_50 = []
    falsealarms_50 = []
    correctnegatives_50 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=28)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            rmse.append(res[0])

            hits_1.append(res[1])
            misses_1.append(res[2])
            falsealarms_1.append(res[3])
            correctnegatives_1.append(res[4])

            hits_5.append(res[5])
            misses_5.append(res[6])
            falsealarms_5.append(res[7])
            correctnegatives_5.append(res[8])

            hits_10.append(res[9])
            misses_10.append(res[10])
            falsealarms_10.append(res[11])
            correctnegatives_10.append(res[12])

            hits_20.append(res[13])
            misses_20.append(res[14])
            falsealarms_20.append(res[15])
            correctnegatives_20.append(res[16])

            hits_30.append(res[17])
            misses_30.append(res[18])
            falsealarms_30.append(res[19])
            correctnegatives_30.append(res[20])

            hits_50.append(res[21])
            misses_50.append(res[22])
            falsealarms_50.append(res[23])
            correctnegatives_50.append(res[24])
            
            
        print("batch complete")
        print(f"current len of rmse {len(rmse)}")
    
    predsv=np.full([256,256,256], np.nan)
    predsv = xr.DataArray(
                predsv,
                dims=['sample', 'lat', 'lon'],
                name='tp'
            )
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        print(x.shape[0])
#         preds = x.squeeze()
#         print(preds.shape)
        preds = x.numpy()
        truth = y.numpy().squeeze(1)
        ###cal for fss
        predsm = preds
        predsa= np.array([preds, predsm])
        ###

        
#        print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=[ 'sample', 'lat', 'lon'],
                name='tp'
            )
        prednew = xr.DataArray(
                predsa,
                dims=['member','sample', 'lat', 'lon'],
                name='tp'
            )
        

        truth = truth * (ds_max - ds_min) + ds_min


        if tp_log:
            truth = log_retrans(truth, tp_log)
        
        mean_fss_5_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=32, device=device)
        preds_fss_5_window32.append(mean_fss_5_window32)
        mean_fss_10_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=32, device=device)
        preds_fss_10_window32.append(mean_fss_10_window32)
        mean_fss_20_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=32, device=device)
        preds_fss_20_window32.append(mean_fss_20_window32)
        mean_fss_30_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=32, device=device)
        preds_fss_30_window32.append(mean_fss_30_window32)
        mean_fss_50_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=32, device=device)
        preds_fss_50_window32.append(mean_fss_50_window32)

        mean_fss_5_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=16, device=device)
        preds_fss_5_window16.append(mean_fss_5_window16)
        mean_fss_10_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=16, device=device)
        preds_fss_10_window16.append(mean_fss_10_window16)
        mean_fss_20_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=16, device=device)
        preds_fss_20_window16.append(mean_fss_20_window16)
        mean_fss_30_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=16, device=device)
        preds_fss_30_window16.append(mean_fss_30_window16)
        mean_fss_50_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=16, device=device)
        preds_fss_50_window16.append(mean_fss_50_window16)

        mean_fss_5_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=8, device=device)
        preds_fss_5_window8.append(mean_fss_5_window8)
        mean_fss_10_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=8, device=device)
        preds_fss_10_window8.append(mean_fss_10_window8)
        mean_fss_20_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=8, device=device)
        preds_fss_20_window8.append(mean_fss_20_window8)
        mean_fss_30_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=8, device=device)
        preds_fss_30_window8.append(mean_fss_30_window8)
        mean_fss_50_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=8, device=device)
        preds_fss_50_window8.append(mean_fss_50_window8)

        mean_fss_5_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=4, device=device)
        preds_fss_5_window4.append(mean_fss_5_window4)
        mean_fss_10_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=4, device=device)
        preds_fss_10_window4.append(mean_fss_10_window4)
        mean_fss_20_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=4, device=device)
        preds_fss_20_window4.append(mean_fss_20_window4)
        mean_fss_30_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=4, device=device)
        preds_fss_30_window4.append(mean_fss_30_window4)
        mean_fss_50_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=4, device=device)
        preds_fss_50_window4.append(mean_fss_50_window4)

        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds))
        pred_hists.append(np.histogram(preds, bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        


        predsv=xr.concat([predsv,preds],dim='sample') 

        pool.starmap_async(compute_metrics_grapes, [(truth, preds, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        
    predsv=predsv[256:,:,:]
    ##writeout data
    predsv[:,:,:].to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/grapes-LT18H.nc')

    
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")

    ts_fenmu_sample=np.array(hits_1)+np.array(misses_1)+np.array(falsealarms_1)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_1)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_1=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_5)+np.array(misses_5)+np.array(falsealarms_5)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_5)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_5=np.mean(ts_all)   ####ts patchmean member 0 50mm
    
    ts_fenmu_sample=np.array(hits_10)+np.array(misses_10)+np.array(falsealarms_10)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_10)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_10=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_20)+np.array(misses_20)+np.array(falsealarms_20)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_20)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_20=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_30)+np.array(misses_30)+np.array(falsealarms_30)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_30)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_30=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_50)+np.array(misses_50)+np.array(falsealarms_50)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_50)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_50=np.mean(ts_all)   ####ts patchmean member 0 50mm
    
    metrics = {"rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 

               "ts_patchmean_1": ts_pm_1,
               "ts_patchmean_5": ts_pm_5,
               "ts_patchmean_10": ts_pm_10,
               "ts_patchmean_20": ts_pm_20,
               "ts_patchmean_30": ts_pm_30,
               "ts_patchmean_50": ts_pm_50,

               "fss_5_window8": np.nanmean(preds_fss_5_window8),
               "fss_10_window8": np.nanmean(preds_fss_10_window8),
               "fss_20_window8": np.nanmean(preds_fss_20_window8),
               "fss_30_window8": np.nanmean(preds_fss_30_window8),
               "fss_50_window8": np.nanmean(preds_fss_50_window8),

               "fss_5_window4": np.nanmean(preds_fss_5_window4),
               "fss_10_window4": np.nanmean(preds_fss_10_window4),
               "fss_20_window4": np.nanmean(preds_fss_20_window4),
               "fss_30_window4": np.nanmean(preds_fss_30_window4),
               "fss_50_window4": np.nanmean(preds_fss_50_window4),

               "fss_5_window16": np.nanmean(preds_fss_5_window16),
               "fss_10_window16": np.nanmean(preds_fss_10_window16),
               "fss_20_window16": np.nanmean(preds_fss_20_window16),
               "fss_30_window16": np.nanmean(preds_fss_30_window16),
               "fss_50_window16": np.nanmean(preds_fss_50_window16),

               "fss_5_window32": np.nanmean(preds_fss_5_window32),
               "fss_10_window32": np.nanmean(preds_fss_10_window32),
               "fss_20_window32": np.nanmean(preds_fss_20_window32),
               "fss_30_window32": np.nanmean(preds_fss_30_window32),
               "fss_50_window32": np.nanmean(preds_fss_50_window32),

                "CSI_1":np.sum(hits_1) / (np.sum(hits_1) + np.sum(falsealarms_1) + np.sum(misses_1)),
                "CSI_5":np.sum(hits_5) / (np.sum(hits_5) + np.sum(falsealarms_5) + np.sum(misses_5)),
                "CSI_10":np.sum(hits_10) / (np.sum(hits_10) + np.sum(falsealarms_10) + np.sum(misses_10)),
                "CSI_20":np.sum(hits_20) / (np.sum(hits_20) + np.sum(falsealarms_20) + np.sum(misses_20)),
                "CSI_30":np.sum(hits_30) / (np.sum(hits_30) + np.sum(falsealarms_30) + np.sum(misses_30)),
                "CSI_50":np.sum(hits_50) / (np.sum(hits_50) + np.sum(falsealarms_50) + np.sum(misses_50)),

                "BIAS_1":(np.sum(hits_1)+np.sum(falsealarms_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "BIAS_5":(np.sum(hits_5)+np.sum(falsealarms_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "BIAS_10":(np.sum(hits_10)+np.sum(falsealarms_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "BIAS_20":(np.sum(hits_20)+np.sum(falsealarms_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "BIAS_30":(np.sum(hits_30)+np.sum(falsealarms_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "BIAS_50":(np.sum(hits_50)+np.sum(falsealarms_50)) / (np.sum(hits_50) +  np.sum(misses_50)),

                "Precision_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(falsealarms_1)),
                "Precision_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(falsealarms_5)),
                "Precision_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(falsealarms_10)),
                "Precision_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(falsealarms_20)),
                "Precision_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(falsealarms_30)),
                "Precision_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(falsealarms_50)),

                "Recall_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "Recall_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "Recall_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "Recall_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "Recall_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "Recall_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(misses_50)),
              }
    
    
    return metrics
###end grapes

###ecmwfifs verification
def compute_metrics_ecmwf(truth, preds, truth_pert, preds_pert, sample):    ##grapes评分计算
    sample_rmse = xs.rmse(preds.sel(sample=sample), truth.sel(sample=sample), dim=['lat', 'lon']).values
    sample_rmse = xs.rmse(preds.sel(sample=sample), truth.sel(sample=sample), dim=['lat', 'lon']).values

    hits_1, misses_1, falsealarms_1, correctnegatives_1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),1)
    hits_5, misses_5, falsealarms_5, correctnegatives_5=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),5)
    hits_10, misses_10, falsealarms_10, correctnegatives_10=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),10)    
    hits_20, misses_20, falsealarms_20, correctnegatives_20=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),20)
    hits_30, misses_30, falsealarms_30, correctnegatives_30=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),30)
    hits_50, misses_50, falsealarms_50, correctnegatives_50=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample),50)
    
    
    return (sample_rmse, 
            hits_1, misses_1, falsealarms_1, correctnegatives_1,
            hits_5, misses_5, falsealarms_5, correctnegatives_5,
            hits_10, misses_10, falsealarms_10, correctnegatives_10,
             hits_20, misses_20, falsealarms_20, correctnegatives_20,
             hits_30, misses_30, falsealarms_30, correctnegatives_30,
             hits_50, misses_50, falsealarms_50, correctnegatives_50)


def ecmwf_interp_patch_eval(dl_test, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    t = TicToc()
  
    rmse = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []

    preds_fss_5_window4 = []
    preds_fss_10_window4 = []
    preds_fss_20_window4 = []
    preds_fss_30_window4 = []
    preds_fss_50_window4 = []

    preds_fss_5_window8 = []
    preds_fss_10_window8 = []
    preds_fss_20_window8 = []
    preds_fss_30_window8 = []
    preds_fss_50_window8 = []
    
    preds_fss_5_window16 = []
    preds_fss_10_window16 = []
    preds_fss_20_window16 = []
    preds_fss_30_window16 = []
    preds_fss_50_window16 = []

    preds_fss_5_window32 = []
    preds_fss_10_window32 = []
    preds_fss_20_window32 = []
    preds_fss_30_window32 = []
    preds_fss_50_window32 = []

    hits_1 = []
    misses_1 = []
    falsealarms_1 = []
    correctnegatives_1 = []

    hits_5 = []
    misses_5 = []
    falsealarms_5 = []
    correctnegatives_5 = []

    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []

    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []

    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []

    hits_50 = []
    misses_50 = []
    falsealarms_50 = []
    correctnegatives_50 = []
    
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=28)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            rmse.append(res[0])
            hits_1.append(res[1])
            misses_1.append(res[2])
            falsealarms_1.append(res[3])
            correctnegatives_1.append(res[4])

            hits_5.append(res[5])
            misses_5.append(res[6])
            falsealarms_5.append(res[7])
            correctnegatives_5.append(res[8])

            hits_10.append(res[9])
            misses_10.append(res[10])
            falsealarms_10.append(res[11])
            correctnegatives_10.append(res[12])

            hits_20.append(res[13])
            misses_20.append(res[14])
            falsealarms_20.append(res[15])
            correctnegatives_20.append(res[16])

            hits_30.append(res[17])
            misses_30.append(res[18])
            falsealarms_30.append(res[19])
            correctnegatives_30.append(res[20])

            hits_50.append(res[21])
            misses_50.append(res[22])
            falsealarms_50.append(res[23])
            correctnegatives_50.append(res[24])

            
        print("batch complete")
        print(f"current len of rmse {len(rmse)}")

    predsv=np.full([256,256,256], np.nan)
    predsv = xr.DataArray(
                predsv,
                dims=['sample', 'lat', 'lon'],
                name='tp'
            )
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        print(x.shape)
        preds = F.interpolate(x, size = (256, 256), mode='bilinear').detach().to('cpu').numpy().squeeze()
        print(preds.shape)
        truth = y.numpy().squeeze(1)
        predsm = preds
        predsa= np.array([preds, predsm])

        
        print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['sample', 'lat', 'lon'],
                name='tp'
            )
        
        prednew = xr.DataArray(
                predsa,
                dims=['member','sample', 'lat', 'lon'],
                name='tp'
            )
        #print(preds)
        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        prednew = prednew * (ds_max - ds_min) + ds_min



        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
            prednew = log_retrans(prednew, tp_log)

        
        mean_fss_5_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=32, device=device)
        preds_fss_5_window32.append(mean_fss_5_window32)
        mean_fss_10_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=32, device=device)
        preds_fss_10_window32.append(mean_fss_10_window32)
        mean_fss_20_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=32, device=device)
        preds_fss_20_window32.append(mean_fss_20_window32)
        mean_fss_30_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=32, device=device)
        preds_fss_30_window32.append(mean_fss_30_window32)
        mean_fss_50_window32 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=32, device=device)
        preds_fss_50_window32.append(mean_fss_50_window32)

        mean_fss_5_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=16, device=device)
        preds_fss_5_window16.append(mean_fss_5_window16)
        mean_fss_10_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=16, device=device)
        preds_fss_10_window16.append(mean_fss_10_window16)
        mean_fss_20_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=16, device=device)
        preds_fss_20_window16.append(mean_fss_20_window16)
        mean_fss_30_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=16, device=device)
        preds_fss_30_window16.append(mean_fss_30_window16)
        mean_fss_50_window16 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=16, device=device)
        preds_fss_50_window16.append(mean_fss_50_window16)

        mean_fss_5_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=8, device=device)
        preds_fss_5_window8.append(mean_fss_5_window8)
        mean_fss_10_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=8, device=device)
        preds_fss_10_window8.append(mean_fss_10_window8)
        mean_fss_20_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=8, device=device)
        preds_fss_20_window8.append(mean_fss_20_window8)
        mean_fss_30_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=8, device=device)
        preds_fss_30_window8.append(mean_fss_30_window8)
        mean_fss_50_window8 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=8, device=device)
        preds_fss_50_window8.append(mean_fss_50_window8)

        mean_fss_5_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=4, device=device)
        preds_fss_5_window4.append(mean_fss_5_window4)
        mean_fss_10_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=4, device=device)
        preds_fss_10_window4.append(mean_fss_10_window4)
        mean_fss_20_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=4, device=device)
        preds_fss_20_window4.append(mean_fss_20_window4)
        mean_fss_30_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=4, device=device)
        preds_fss_30_window4.append(mean_fss_30_window4)
        mean_fss_50_window4 = fss(prednew.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 50, window=4, device=device)
        preds_fss_50_window4.append(mean_fss_50_window4)

        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds))
        pred_hists.append(np.histogram(preds, bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape)

        predsv=xr.concat([predsv,preds],dim='sample') 

        pool.starmap_async(compute_metrics_ecmwf, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        
    predsv=predsv[256:,:,:]
    ##writeout data
    predsv[:,:,:].to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/ecmwf-LT18H.nc')

    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")

    ts_fenmu_sample=np.array(hits_1)+np.array(misses_1)+np.array(falsealarms_1)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_1)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_1=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_5)+np.array(misses_5)+np.array(falsealarms_5)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_5)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_5=np.mean(ts_all)   ####ts patchmean member 0 50mm
    
    ts_fenmu_sample=np.array(hits_10)+np.array(misses_10)+np.array(falsealarms_10)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_10)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_10=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_20)+np.array(misses_20)+np.array(falsealarms_20)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_20)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_20=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_30)+np.array(misses_30)+np.array(falsealarms_30)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_30)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_30=np.mean(ts_all)   ####ts patchmean member 0 50mm

    ts_fenmu_sample=np.array(hits_50)+np.array(misses_50)+np.array(falsealarms_50)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_50)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_50=np.mean(ts_all)   ####ts patchmean member 0 50mm

    
    metrics = {"rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 

               "ts_patchmean_1": ts_pm_1,
               "ts_patchmean_5": ts_pm_5,
               "ts_patchmean_10": ts_pm_10,
               "ts_patchmean_20": ts_pm_20,
               "ts_patchmean_30": ts_pm_30,
               "ts_patchmean_50": ts_pm_50,
               
               "fss_5_window8": np.nanmean(preds_fss_5_window8),
               "fss_10_window8": np.nanmean(preds_fss_10_window8),
               "fss_20_window8": np.nanmean(preds_fss_20_window8),
               "fss_30_window8": np.nanmean(preds_fss_30_window8),
               "fss_50_window8": np.nanmean(preds_fss_50_window8),

               "fss_5_window4": np.nanmean(preds_fss_5_window4),
               "fss_10_window4": np.nanmean(preds_fss_10_window4),
               "fss_20_window4": np.nanmean(preds_fss_20_window4),
               "fss_30_window4": np.nanmean(preds_fss_30_window4),
               "fss_50_window4": np.nanmean(preds_fss_50_window4),

               "fss_5_window16": np.nanmean(preds_fss_5_window16),
               "fss_10_window16": np.nanmean(preds_fss_10_window16),
               "fss_20_window16": np.nanmean(preds_fss_20_window16),
               "fss_30_window16": np.nanmean(preds_fss_30_window16),
               "fss_50_window16": np.nanmean(preds_fss_50_window16),

               "fss_5_window32": np.nanmean(preds_fss_5_window32),
               "fss_10_window32": np.nanmean(preds_fss_10_window32),
               "fss_20_window32": np.nanmean(preds_fss_20_window32),
               "fss_30_window32": np.nanmean(preds_fss_30_window32),
               "fss_50_window32": np.nanmean(preds_fss_50_window32),

               "CSI_1":np.sum(hits_1) / (np.sum(hits_1) + np.sum(falsealarms_1) + np.sum(misses_1)),
                "CSI_5":np.sum(hits_5) / (np.sum(hits_5) + np.sum(falsealarms_5) + np.sum(misses_5)),
                "CSI_10":np.sum(hits_10) / (np.sum(hits_10) + np.sum(falsealarms_10) + np.sum(misses_10)),
                "CSI_20":np.sum(hits_20) / (np.sum(hits_20) + np.sum(falsealarms_20) + np.sum(misses_20)),
                "CSI_30":np.sum(hits_30) / (np.sum(hits_30) + np.sum(falsealarms_30) + np.sum(misses_30)),
                "CSI_50":np.sum(hits_50) / (np.sum(hits_50) + np.sum(falsealarms_50) + np.sum(misses_50)),

                "BIAS_1":(np.sum(hits_1)+np.sum(falsealarms_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "BIAS_5":(np.sum(hits_5)+np.sum(falsealarms_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "BIAS_10":(np.sum(hits_10)+np.sum(falsealarms_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "BIAS_20":(np.sum(hits_20)+np.sum(falsealarms_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "BIAS_30":(np.sum(hits_30)+np.sum(falsealarms_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "BIAS_50":(np.sum(hits_50)+np.sum(falsealarms_50)) / (np.sum(hits_50) +  np.sum(misses_50)),

                "Precision_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(falsealarms_1)),
                "Precision_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(falsealarms_5)),
                "Precision_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(falsealarms_10)),
                "Precision_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(falsealarms_20)),
                "Precision_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(falsealarms_30)),
                "Precision_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(falsealarms_50)),

                "Recall_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "Recall_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "Recall_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "Recall_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "Recall_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "Recall_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(misses_50)),

              }
    
    
    return metrics

###end ecmwf verfication




def par_gen_patch_eval_junxu_forroc(gen, dl_test, nens, ds_min, ds_max, tp_log, device):    ##cgan评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    rhist = []
    rels_20 = []
    rels_30 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_10 = []
    preds_brier_20 = []
    preds_brier_30 = []
    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []
    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []
    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []
    roc=[]
    
    #t.tic()
    #num_workers = mp.cpu_count()
    #print("num_workers:", num_workers)
    #pool = Pool(processes=num_workers)
    #t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            rmse.append(res[1])
            rhist.append(res[2])
            rels_20.append(res[3])
            rels_30.append(res[4])
            preds_brier_10.append(res[5])
            preds_brier_20.append(res[6])
            preds_brier_30.append(res[7])
            hits_10.append(res[8])
            misses_10.append(res[9])
            falsealarms_10.append(res[10])
            correctnegatives_10.append(res[11])
            hits_20.append(res[12])
            misses_20.append(res[13])
            falsealarms_20.append(res[14])
            correctnegatives_20.append(res[15])
            hits_30.append(res[16])
            misses_30.append(res[17])
            falsealarms_30.append(res[18])
            correctnegatives_30.append(res[19])


            
        print("batch complete")
        print(f"current len of crps {len(crps)}")

    truthnew=np.full([256, 256,256], np.nan)
    prednew=np.full([10,256,256,256], np.nan)
    
    truthnew= xr.DataArray(
                truthnew,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
    prednew = xr.DataArray(
                prednew,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )
            
    for batch_idx, (x,y,href) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)

        truthnew=xr.concat([truthnew,truth],dim='sample')
        prednew=xr.concat([prednew,preds],dim='sample')
        print(truthnew.shape)
        print(prednew.shape)
        
    print('this is 20mm')
    print(truthnew[0:255,:,:])
    prednew=prednew[:,256:,:,:]
    truthnew=truthnew[256:,:,:]
    print(prednew)
    print(truthnew)
    pscore=(prednew > 20.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 20.0).values.reshape(-1)

    auc=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    print(auc)

    prob_true, prob_pred=calibration_curve(ytruth, pscore,  n_bins=10)
    print(prob_true)
    print(prob_pred)


    fpr, tpr, thresholds = metrics.roc_curve(ytruth, pscore)
    print(fpr)
    print(tpr)
    print(thresholds)


    print('this is 10mm')
    pscore=(prednew > 10.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 10.0).values.reshape(-1)

    auc=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    print(auc)

    prob_true, prob_pred=calibration_curve(ytruth, pscore,  n_bins=11)
    print(prob_true)
    print(prob_pred)


    fpr, tpr, thresholds = metrics.roc_curve(ytruth, pscore)
    print(fpr)
    print(tpr)
    print(thresholds)


 
    return roc


###CGAN verification- all scores 
def compute_metrics_all(truth, preds, truth_pert, preds_pert, sample):    ##CGAN评分计算
    sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
    truth_course = truth.coarsen(lat=4, lon=4)
    preds_course = preds.coarsen(lat=4, lon=4)
    sample_max_pool_crps = xs.crps_ensemble(truth_course.max().sel(sample=sample), preds_course.max().sel(sample=sample)).values
    sample_avg_pool_crps = xs.crps_ensemble(truth_course.mean().sel(sample=sample), preds_course.mean().sel(sample=sample)).values
    sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
    rhist = xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values

    rel1 = xs.reliability(truth.sel(sample=sample)>1.0,(preds.sel(sample=sample)>1.0).mean('member'))
    rel1 = xr.where(np.isnan(rel1), 0, rel1)
    rel1['relative_freq'] = rel1

    rel5 = xs.reliability(truth.sel(sample=sample)>5.0,(preds.sel(sample=sample)>5.0).mean('member'))
    rel5 = xr.where(np.isnan(rel5), 0, rel5)
    rel5['relative_freq'] = rel5

    rel10 = xs.reliability(truth.sel(sample=sample)>10,(preds.sel(sample=sample)>10).mean('member'))
    rel10 = xr.where(np.isnan(rel10), 0, rel10)
    rel10['relative_freq'] = rel10
    
    rel20 = xs.reliability(truth.sel(sample=sample)>20,(preds.sel(sample=sample)>20).mean('member'))
    rel20 = xr.where(np.isnan(rel20), 0, rel20)
    rel20['relative_freq'] = rel20
    
    rel30 = xs.reliability(truth.sel(sample=sample)>30,(preds.sel(sample=sample)>30).mean('member'))
    rel30 = xr.where(np.isnan(rel30), 0, rel30)
    rel30['relative_freq'] = rel30

    sample_brier_1 = xs.brier_score(truth.sel(sample=sample) > 1.0, (preds.sel(sample=sample) > 1.0).mean('member'), dim=['lat', 'lon'])
    sample_brier_5 = xs.brier_score(truth.sel(sample=sample) > 5.0, (preds.sel(sample=sample) > 5.0).mean('member'), dim=['lat', 'lon'])
    sample_brier_10 = xs.brier_score(truth.sel(sample=sample) > 10.0, (preds.sel(sample=sample) > 10.0).mean('member'), dim=['lat', 'lon'])
    sample_brier_20 = xs.brier_score(truth.sel(sample=sample) > 20.0, (preds.sel(sample=sample) > 20.0).mean('member'), dim=['lat', 'lon'])
    sample_brier_30 = xs.brier_score(truth.sel(sample=sample) > 30.0, (preds.sel(sample=sample) > 30.0).mean('member'), dim=['lat', 'lon'])
    
    hits_1, misses_1, falsealarms_1, correctnegatives_1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),1)
    hits_5, misses_5, falsealarms_5, correctnegatives_5=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),5)
    hits_10, misses_10, falsealarms_10, correctnegatives_10=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),10)
    hits_20, misses_20, falsealarms_20, correctnegatives_20=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),20)
    hits_30, misses_30, falsealarms_30, correctnegatives_30=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),30)
    hits_50, misses_50, falsealarms_50, correctnegatives_50=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample).mean('member'),50)
    
    hits_1_m0, misses_1_m0, falsealarms_1_m0, correctnegatives_1_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),1)
    hits_5_m0, misses_5_m0, falsealarms_5_m0, correctnegatives_5_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),5)
    hits_10_m0, misses_10_m0, falsealarms_10_m0, correctnegatives_10_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),10)
    hits_20_m0, misses_20_m0, falsealarms_20_m0, correctnegatives_20_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),20)
    hits_30_m0, misses_30_m0, falsealarms_30_m0, correctnegatives_30_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),30)
    hits_50_m0, misses_50_m0, falsealarms_50_m0, correctnegatives_50_m0=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=0),50)

    hits_1_m1, misses_1_m1, falsealarms_1_m1, correctnegatives_1_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),1)
    hits_5_m1, misses_5_m1, falsealarms_5_m1, correctnegatives_5_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),5)
    hits_10_m1, misses_10_m1, falsealarms_10_m1, correctnegatives_10_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),10)
    hits_20_m1, misses_20_m1, falsealarms_20_m1, correctnegatives_20_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),20)
    hits_30_m1, misses_30_m1, falsealarms_30_m1, correctnegatives_30_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),30)
    hits_50_m1, misses_50_m1, falsealarms_50_m1, correctnegatives_50_m1=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=1),50)

    hits_1_m2, misses_1_m2, falsealarms_1_m2, correctnegatives_1_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),1)
    hits_5_m2, misses_5_m2, falsealarms_5_m2, correctnegatives_5_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),5)
    hits_10_m2, misses_10_m2, falsealarms_10_m2, correctnegatives_10_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),10)
    hits_20_m2, misses_20_m2, falsealarms_20_m2, correctnegatives_20_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),20)
    hits_30_m2, misses_30_m2, falsealarms_30_m2, correctnegatives_30_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),30)
    hits_50_m2, misses_50_m2, falsealarms_50_m2, correctnegatives_50_m2=prep_clf(truth.sel(sample=sample), preds.sel(sample=sample,member=2),50)


    
    return (sample_crps,  sample_max_pool_crps, sample_avg_pool_crps, sample_rmse,  rhist, 
            rel1, rel5, rel10, rel20, rel30, 
            sample_brier_1, sample_brier_5, sample_brier_10, sample_brier_20, sample_brier_30,

            hits_1, misses_1, falsealarms_1, correctnegatives_1,
            hits_5, misses_5, falsealarms_5, correctnegatives_5,
            hits_10, misses_10, falsealarms_10, correctnegatives_10,
            hits_20, misses_20, falsealarms_20, correctnegatives_20,
            hits_30, misses_30, falsealarms_30, correctnegatives_30,
            hits_50, misses_50, falsealarms_50, correctnegatives_50,

            hits_1_m0, misses_1_m0, falsealarms_1_m0, correctnegatives_1_m0,
            hits_5_m0, misses_5_m0, falsealarms_5_m0, correctnegatives_5_m0,
            hits_10_m0, misses_10_m0, falsealarms_10_m0, correctnegatives_10_m0,
            hits_20_m0, misses_20_m0, falsealarms_20_m0, correctnegatives_20_m0,
            hits_30_m0, misses_30_m0, falsealarms_30_m0, correctnegatives_30_m0,
            hits_50_m0, misses_50_m0, falsealarms_50_m0, correctnegatives_50_m0,
            
            hits_1_m1, misses_1_m1, falsealarms_1_m1, correctnegatives_1_m1,
            hits_5_m1, misses_5_m1, falsealarms_5_m1, correctnegatives_5_m1,
            hits_10_m1, misses_10_m1, falsealarms_10_m1, correctnegatives_10_m1,
            hits_20_m1, misses_20_m1, falsealarms_20_m1, correctnegatives_20_m1,
            hits_30_m1, misses_30_m1, falsealarms_30_m1, correctnegatives_30_m1,
            hits_50_m1, misses_50_m1, falsealarms_50_m1, correctnegatives_50_m1,
            
            hits_1_m2, misses_1_m2, falsealarms_1_m2, correctnegatives_1_m2,
            hits_5_m2, misses_5_m2, falsealarms_5_m2, correctnegatives_5_m2,
            hits_10_m2, misses_10_m2, falsealarms_10_m2, correctnegatives_10_m2,
            hits_20_m2, misses_20_m2, falsealarms_20_m2, correctnegatives_20_m2,
            hits_30_m2, misses_30_m2, falsealarms_30_m2, correctnegatives_30_m2,
            hits_50_m2, misses_50_m2, falsealarms_50_m2, correctnegatives_50_m2)



def par_gen_patch_eval_all(gen, dl_test, nens, ds_min, ds_max, tp_log, pred_ncname, device):    ##cgan评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    max_pool_crps = []
    avg_pool_crps = []

    rmse = []
    rhist = []

    ###reliability 1, 5, 10, 20, 30 
    rels_1 = []
    rels_5 = []
    rels_10 = []
    rels_20 = []
    rels_30 = []

    ###calibrition curve
    prob_true_1= [] 
    prob_pred_1 =[]
    prob_true_5= [] 
    prob_pred_5 =[]
    prob_true_10= [] 
    prob_pred_10 =[]
    prob_true_20= [] 
    prob_pred_20 =[]
    prob_true_30= [] 
    prob_pred_30 =[]

    ####ROC 1, 5, 10, 20, 30
    auc_1 = []
    auc_5 = []
    auc_10 = []
    auc_20 = []
    auc_30 = []
    
    fpr_1 = []
    tpr_1 = []
    thresholds_1 = []
    fpr_5 = []
    tpr_5 = []
    thresholds_5 = []
    fpr_10 = []
    tpr_10 = []
    thresholds_10 = []
    fpr_20 = []
    tpr_20 = []
    thresholds_20 = []
    fpr_30 = []
    tpr_30 = []
    thresholds_30 = []
    

    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []

    pred_means_m1 = []
    pred_hists_m1 = []


    
    pred_means_m2 = []
    pred_hists_m2 = []


    ###FSS thresthholds windows
    preds_fss_5_window4 = []
    preds_fss_10_window4 = []    ###default 10mm window 4km
    preds_fss_20_window4 = []
    preds_fss_30_window4 = []

    preds_fss_5_window8 = []
    preds_fss_10_window8 = []    ###default 10mm window 10km
    preds_fss_20_window8 = []
    preds_fss_30_window8 = []

    preds_fss_5_window16 = []
    preds_fss_10_window16 = []    ###default 10mm window 20km
    preds_fss_20_window16 = []
    preds_fss_30_window16 = []

    preds_fss_5_window16 = []
    preds_fss_10_window16 = []    ###default 10mm window 20km
    preds_fss_20_window16 = []
    preds_fss_30_window16 = []

    preds_fss_5_window32 = []
    preds_fss_10_window32 = []    ###default 10mm window 20km
    preds_fss_20_window32 = []
    preds_fss_30_window32 = []



    preds_brier_1 = []   ###brier score
    preds_brier_5 = []
    preds_brier_10 = []
    preds_brier_20 = []
    preds_brier_30 = []

    hits_1 = []
    misses_1 = []
    falsealarms_1 = []
    correctnegatives_1 = []
    
    hits_5 = []
    misses_5 = []
    falsealarms_5 = []
    correctnegatives_5 = []

    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []

    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []

    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []

    hits_50 = []
    misses_50 = []
    falsealarms_50 = []
    correctnegatives_50 = []

    hits_1_m0 = []
    misses_1_m0 = []
    falsealarms_1_m0 = []
    correctnegatives_1_m0 = []
    
    hits_5_m0 = []
    misses_5_m0 = []
    falsealarms_5_m0 = []
    correctnegatives_5_m0 = []

    hits_10_m0 = []
    misses_10_m0 = []
    falsealarms_10_m0 = []
    correctnegatives_10_m0 = []

    hits_20_m0 = []
    misses_20_m0 =[]
    falsealarms_20_m0 = []
    correctnegatives_20_m0 = []

    hits_30_m0 = []
    misses_30_m0 = []
    falsealarms_30_m0 = []
    correctnegatives_30_m0 = []

    hits_50_m0 = []
    misses_50_m0 = []
    falsealarms_50_m0 = []
    correctnegatives_50_m0 = []

    hits_1_m1 = []
    misses_1_m1 = []
    falsealarms_1_m1 = []
    correctnegatives_1_m1 = []
    
    hits_5_m1 = []
    misses_5_m1 = []
    falsealarms_5_m1 = []
    correctnegatives_5_m1 = []

    hits_10_m1 = []
    misses_10_m1 = []
    falsealarms_10_m1 = []
    correctnegatives_10_m1 = []

    hits_20_m1 = []
    misses_20_m1 =[]
    falsealarms_20_m1 = []
    correctnegatives_20_m1 = []

    hits_30_m1 = []
    misses_30_m1 = []
    falsealarms_30_m1 = []
    correctnegatives_30_m1 = []

    hits_50_m1 = []
    misses_50_m1 = []
    falsealarms_50_m1 = []
    correctnegatives_50_m1 = []

    hits_1_m2 = []
    misses_1_m2 = []
    falsealarms_1_m2 = []
    correctnegatives_1_m2 = []
    
    hits_5_m2 = []
    misses_5_m2 = []
    falsealarms_5_m2 = []
    correctnegatives_5_m2 = []

    hits_10_m2 = []
    misses_10_m2 = []
    falsealarms_10_m2 = []
    correctnegatives_10_m2 = []

    hits_20_m2 = []
    misses_20_m2 =[]
    falsealarms_20_m2 = []
    correctnegatives_20_m2 = []

    hits_30_m2 = []
    misses_30_m2 = []
    falsealarms_30_m2 = []
    correctnegatives_30_m2 = []

    hits_50_m2 = []
    misses_50_m2 = []
    falsealarms_50_m2 = []
    correctnegatives_50_m2 = []




    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=28)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:

            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])

            rels_1.append(res[5])
            rels_5.append(res[6])
            rels_10.append(res[7])
            rels_20.append(res[8])
            rels_30.append(res[9])

            preds_brier_1.append(res[10])
            preds_brier_5.append(res[11])
            preds_brier_10.append(res[12])
            preds_brier_20.append(res[13])
            preds_brier_30.append(res[14])

            hits_1.append(res[15])
            misses_1.append(res[16])
            falsealarms_1.append(res[17])
            correctnegatives_1.append(res[18])

            hits_5.append(res[19])
            misses_5.append(res[20])
            falsealarms_5.append(res[21])
            correctnegatives_5.append(res[22])

            hits_10.append(res[23])
            misses_10.append(res[24])
            falsealarms_10.append(res[25])
            correctnegatives_10.append(res[26])

            hits_20.append(res[27])
            misses_20.append(res[28])
            falsealarms_20.append(res[29])
            correctnegatives_20.append(res[30])

            hits_30.append(res[31])
            misses_30.append(res[32])
            falsealarms_30.append(res[33])
            correctnegatives_30.append(res[34])

            hits_50.append(res[35])
            misses_50.append(res[36])
            falsealarms_50.append(res[37])
            correctnegatives_50.append(res[38])

            hits_1_m0.append(res[39])
            misses_1_m0.append(res[40])
            falsealarms_1_m0.append(res[41])
            correctnegatives_1_m0.append(res[42])

            hits_5_m0.append(res[43])
            misses_5_m0.append(res[44])
            falsealarms_5_m0.append(res[45])
            correctnegatives_5_m0.append(res[46])

            hits_10_m0.append(res[47])
            misses_10_m0.append(res[48])
            falsealarms_10_m0.append(res[49])
            correctnegatives_10_m0.append(res[50])

            hits_20_m0.append(res[51])
            misses_20_m0.append(res[52])
            falsealarms_20_m0.append(res[53])
            correctnegatives_20_m0.append(res[54])

            hits_30_m0.append(res[55])
            misses_30_m0.append(res[56])
            falsealarms_30_m0.append(res[57])
            correctnegatives_30_m0.append(res[58])

            hits_50_m0.append(res[59])
            misses_50_m0.append(res[60])
            falsealarms_50_m0.append(res[61])
            correctnegatives_50_m0.append(res[62])

            hits_1_m1.append(res[63])
            misses_1_m1.append(res[64])
            falsealarms_1_m1.append(res[65])
            correctnegatives_1_m1.append(res[66])

            hits_5_m1.append(res[67])
            misses_5_m1.append(res[68])
            falsealarms_5_m1.append(res[69])
            correctnegatives_5_m1.append(res[70])

            hits_10_m1.append(res[71])
            misses_10_m1.append(res[72])
            falsealarms_10_m1.append(res[73])
            correctnegatives_10_m1.append(res[74])

            hits_20_m1.append(res[75])
            misses_20_m1.append(res[76])
            falsealarms_20_m1.append(res[77])
            correctnegatives_20_m1.append(res[78])

            hits_30_m1.append(res[79])
            misses_30_m1.append(res[80])
            falsealarms_30_m1.append(res[81])
            correctnegatives_30_m1.append(res[82])

            hits_50_m1.append(res[83])
            misses_50_m1.append(res[84])
            falsealarms_50_m1.append(res[85])
            correctnegatives_50_m1.append(res[86])

            hits_1_m2.append(res[87])
            misses_1_m2.append(res[88])
            falsealarms_1_m2.append(res[89])
            correctnegatives_1_m2.append(res[90])

            hits_5_m2.append(res[91])
            misses_5_m2.append(res[92])
            falsealarms_5_m2.append(res[93])
            correctnegatives_5_m2.append(res[94])

            hits_10_m2.append(res[95])
            misses_10_m2.append(res[96])
            falsealarms_10_m2.append(res[97])
            correctnegatives_10_m2.append(res[98])

            hits_20_m2.append(res[99])
            misses_20_m2.append(res[100])
            falsealarms_20_m2.append(res[101])
            correctnegatives_20_m2.append(res[102])

            hits_30_m2.append(res[103])
            misses_30_m2.append(res[104])
            falsealarms_30_m2.append(res[105])
            correctnegatives_30_m2.append(res[106])

            hits_50_m2.append(res[107])
            misses_50_m2.append(res[108])
            falsealarms_50_m2.append(res[109])
            correctnegatives_50_m2.append(res[110])




            
        print("batch complete")
        print(f"current len of crps {len(crps)}")

    truthnew=np.full([256, 256,256], np.nan)
    prednew=np.full([10,256,256,256], np.nan)
    
    truthnew= xr.DataArray(
                truthnew,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
    prednew = xr.DataArray(
                prednew,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )
            
    for batch_idx, (x,y,href) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        

        
        ###FSS
        mean_fss_5_window4 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=4, device=device) ###member平均的FSS
        preds_fss_5_window4.append(mean_fss_5_window4)
        mean_fss_10_window4 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=4, device=device) ###member平均的FSS
        preds_fss_10_window4.append(mean_fss_10_window4)
        mean_fss_20_window4 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=4, device=device)
        preds_fss_20_window4.append(mean_fss_20_window4)
        mean_fss_30_window4 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=4, device=device)
        preds_fss_30_window4.append(mean_fss_30_window4)

        mean_fss_5_window8 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=8, device=device) ###member平均的FSS
        preds_fss_5_window8.append(mean_fss_5_window8)
        mean_fss_10_window8 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=8, device=device) ###member平均的FSS
        preds_fss_10_window8.append(mean_fss_10_window8)
        mean_fss_20_window8 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=8, device=device)
        preds_fss_20_window8.append(mean_fss_20_window8)
        mean_fss_30_window8 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=8, device=device)
        preds_fss_30_window8.append(mean_fss_30_window8)

        mean_fss_5_window16 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=16, device=device) ###member平均的FSS
        preds_fss_5_window16.append(mean_fss_5_window16)
        mean_fss_10_window16 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=16, device=device) ###member平均的FSS
        preds_fss_10_window16.append(mean_fss_10_window16)
        mean_fss_20_window16 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=16, device=device)
        preds_fss_20_window16.append(mean_fss_20_window16)
        mean_fss_30_window16 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=16, device=device)
        preds_fss_30_window16.append(mean_fss_30_window16)

        mean_fss_5_window32 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 5, window=32, device=device) ###member平均的FSS
        preds_fss_5_window32.append(mean_fss_5_window32)
        mean_fss_10_window32 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 10, window=32, device=device) ###member平均的FSS
        preds_fss_10_window32.append(mean_fss_10_window32)
        mean_fss_20_window32 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=32, device=device)
        preds_fss_20_window32.append(mean_fss_20_window32)
        mean_fss_30_window32 = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 30, window=32, device=device)
        preds_fss_30_window32.append(mean_fss_30_window32)





        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0))) ###0member的preds
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0]) ###0member的hist
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])

        pred_means_m1.append(np.mean(preds.sel(member=1))) ###1member的preds
        pred_hists_m1.append(np.histogram(preds.sel(member=1), bins = bin_edges, density=False)[0]) ###0member的hist

        pred_means_m2.append(np.mean(preds.sel(member=2))) ###2member的preds
        pred_hists_m2.append(np.histogram(preds.sel(member=2), bins = bin_edges, density=False)[0]) ###2member的hist

        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape) ##扰动
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) ##扰动
        
        truthnew=xr.concat([truthnew,truth],dim='sample')
        prednew=xr.concat([prednew,preds],dim='sample')


        pool.starmap_async(compute_metrics_all, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')

    ###sklearn ROC Reliability 相关

    prednew=prednew[:,256:,:,:]
    truthnew=truthnew[256:,:,:]

    ##writeout data
    prednew[0,:,:,:].to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/pred-member0-'+ pred_ncname + '-LT012.nc')
    prednew[1,:,:,:].to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/pred-member1-'+ pred_ncname + '-LT012.nc')
    prednew[2,:,:,:].to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/pred-member2-'+ pred_ncname + '-LT012.nc')
    truthnew.to_netcdf('/LOCALDISK/smlzhuhai_jwma_1/evaluation/truthnew'+ pred_ncname + '.nc')
    ###endwriteout

    pscore=(prednew > 1.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 1.0).values.reshape(-1)
    auc_1 = metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    prob_true_1, prob_pred_1=calibration_curve(ytruth, pscore,  n_bins=11)
    fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(ytruth, pscore)

    pscore=(prednew > 5.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 5.0).values.reshape(-1)
    auc_5=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    prob_true_5, prob_pred_5=calibration_curve(ytruth, pscore,  n_bins=11)
    fpr_5, tpr_5, thresholds_5 = metrics.roc_curve(ytruth, pscore)

    pscore=(prednew > 10.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 10.0).values.reshape(-1)
    auc_10=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    prob_true_10, prob_pred_10=calibration_curve(ytruth, pscore,  n_bins=11)
    fpr_10, tpr_10, thresholds_10 = metrics.roc_curve(ytruth, pscore)

    pscore=(prednew > 20.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 20.0).values.reshape(-1)
    auc_20=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    prob_true_20, prob_pred_20=calibration_curve(ytruth, pscore,  n_bins=10)
    fpr_20, tpr_20, thresholds_20 = metrics.roc_curve(ytruth, pscore)
    
    pscore=(prednew > 30.0).mean('member').values.reshape(-1)
    ytruth=(truthnew > 30.0).values.reshape(-1)
    auc_30=metrics.roc_auc_score(ytruth, pscore,  sample_weight=None)
    prob_true_30, prob_pred_30=calibration_curve(ytruth, pscore,  n_bins=10)
    fpr_30, tpr_30, thresholds_30 = metrics.roc_curve(ytruth, pscore)





   ###Reliability   XS  
    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability   

    rels_5 = xr.concat(rels_5, dim = "patch")
    weights_5 = rels_5.samples / rels_5.samples.sum(dim="patch")
    weighted_relative_freq_5 = (weights_5*rels_5.relative_freq).sum(dim="patch")
    samples_5 = rels_5.samples.sum(dim="patch")
    forecast_probs_5 = rels_5.forecast_probability

    rels_10 = xr.concat(rels_10, dim = "patch")
    weights_10 = rels_10.samples / rels_10.samples.sum(dim="patch")
    weighted_relative_freq_10 = (weights_10*rels_10.relative_freq).sum(dim="patch")
    samples_10 = rels_10.samples.sum(dim="patch")
    forecast_probs_10 = rels_10.forecast_probability     

    rels_20 = xr.concat(rels_20, dim = "patch")
    weights_20 = rels_20.samples / rels_20.samples.sum(dim="patch")
    weighted_relative_freq_20 = (weights_20*rels_20.relative_freq).sum(dim="patch")
    samples_20 = rels_20.samples.sum(dim="patch")
    forecast_probs_20 = rels_20.forecast_probability
    
    rels_30 = xr.concat(rels_30, dim = "patch")
    weights_30 = rels_30.samples / rels_30.samples.sum(dim="patch")
    weighted_relative_freq_30 = (weights_30*rels_30.relative_freq).sum(dim="patch")
    samples_30 = rels_30.samples.sum(dim="patch")
    forecast_probs_30 = rels_30.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    pred_hists_m1 = (np.sum(np.array(pred_hists_m1), axis=0), bin_edges)
    pred_hists_m2 = (np.sum(np.array(pred_hists_m2), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)

    ts_fenmu_sample=np.array(hits_1_m0)+np.array(misses_1_m0)+np.array(falsealarms_1_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_1_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_1=np.mean(ts_all)   ####ts patchmean member 0 1mm

    ts_fenmu_sample=np.array(hits_5_m0)+np.array(misses_5_m0)+np.array(falsealarms_5_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_5_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_5=np.mean(ts_all)   ####ts patchmean member 0 5mm

    ts_fenmu_sample=np.array(hits_10_m0)+np.array(misses_10_m0)+np.array(falsealarms_10_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_10_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_10=np.mean(ts_all)   ####ts patchmean member 0 10mm

    ts_fenmu_sample=np.array(hits_20_m0)+np.array(misses_20_m0)+np.array(falsealarms_20_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_20_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_20=np.mean(ts_all)   ####ts patchmean member 0 20mm
    
    ts_fenmu_sample=np.array(hits_30_m0)+np.array(misses_30_m0)+np.array(falsealarms_30_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_30_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_30=np.mean(ts_all)   ####ts patchmean member 0 30mm

    ts_fenmu_sample=np.array(hits_50_m0)+np.array(misses_50_m0)+np.array(falsealarms_50_m0)
    nonzero_ts_fenmu_sample = ts_fenmu_sample != 0
    hitsa=np.array(hits_50_m0)
    ts_all = hitsa[nonzero_ts_fenmu_sample]/ts_fenmu_sample[nonzero_ts_fenmu_sample]
    ts_pm_50=np.mean(ts_all)   ####ts patchmean member 0 50mm



    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metricses = {"crps": np.mean(crps), 
               "rankhist": rhist, 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),

               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_5": (weighted_relative_freq_5, forecast_probs_5, samples_5),
               "reliability_10": (weighted_relative_freq_10, forecast_probs_10, samples_10), 
               "reliability_20": (weighted_relative_freq_20, forecast_probs_20, samples_20), 
               "reliability_30": (weighted_relative_freq_30, forecast_probs_30, samples_30), 

               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean_member0": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist_member0": pred_hists, 

               "preds_mean_member1": np.mean(pred_means_m1), 
               "preds_hist_mamber1": pred_hists_m1, 

               
               "preds_mean_member2": np.mean(pred_means_m2), 
               "preds_hist_mamber2": pred_hists_m2,

               "ts_patchmean_1": ts_pm_1,
               "ts_patchmean_5": ts_pm_5,
               "ts_patchmean_10": ts_pm_10,
               "ts_patchmean_20": ts_pm_20,
               "ts_patchmean_30": ts_pm_30,
               "ts_patchmean_50": ts_pm_50,


               "fss_5_window4": np.mean(preds_fss_5_window4),
               "fss_10_window4": np.mean(preds_fss_10_window4),
               "fss_20_window4":np.mean(preds_fss_20_window4),
               "fss_30_window4":np.mean(preds_fss_30_window4),

               "fss_5_window8": np.mean(preds_fss_5_window8),
               "fss_10_window8": np.mean(preds_fss_10_window8),
               "fss_20_window8":np.mean(preds_fss_20_window8),
               "fss_30_window8":np.mean(preds_fss_30_window8),

               "fss_5_window16": np.mean(preds_fss_5_window16),
               "fss_10_window16": np.mean(preds_fss_10_window16),
               "fss_20_window16":np.mean(preds_fss_20_window16),
               "fss_30_window16":np.mean(preds_fss_30_window16),

               "fss_5_window32": np.mean(preds_fss_5_window32),
               "fss_10_window32": np.mean(preds_fss_10_window32),
               "fss_20_window32":np.mean(preds_fss_20_window32),
               "fss_30_window32":np.mean(preds_fss_30_window32),

               "preds_brier_1" : np.mean(preds_brier_1),
               "preds_brier_5" : np.mean(preds_brier_5),
               "preds_brier_10" : np.mean(preds_brier_10),
               "preds_brier_20" : np.mean(preds_brier_20),
               "preds_brier_30": np.mean(preds_brier_30),

                "CSI_1":np.sum(hits_1) / (np.sum(hits_1) + np.sum(falsealarms_1) + np.sum(misses_1)),
                "CSI_5":np.sum(hits_5) / (np.sum(hits_5) + np.sum(falsealarms_5) + np.sum(misses_5)),
                "CSI_10":np.sum(hits_10) / (np.sum(hits_10) + np.sum(falsealarms_10) + np.sum(misses_10)),
                "CSI_20":np.sum(hits_20) / (np.sum(hits_20) + np.sum(falsealarms_20) + np.sum(misses_20)),
                "CSI_30":np.sum(hits_30) / (np.sum(hits_30) + np.sum(falsealarms_30) + np.sum(misses_30)),
                "CSI_50":np.sum(hits_50) / (np.sum(hits_50) + np.sum(falsealarms_50) + np.sum(misses_50)),

                "BIAS_1":(np.sum(hits_1)+np.sum(falsealarms_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "BIAS_5":(np.sum(hits_5)+np.sum(falsealarms_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "BIAS_10":(np.sum(hits_10)+np.sum(falsealarms_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "BIAS_20":(np.sum(hits_20)+np.sum(falsealarms_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "BIAS_30":(np.sum(hits_30)+np.sum(falsealarms_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "BIAS_50":(np.sum(hits_50)+np.sum(falsealarms_50)) / (np.sum(hits_50) +  np.sum(misses_50)),

                "Precision_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(falsealarms_1)),
                "Precision_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(falsealarms_5)),
                "Precision_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(falsealarms_10)),
                "Precision_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(falsealarms_20)),
                "Precision_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(falsealarms_30)),
                "Precision_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(falsealarms_50)),

                "Recall_1":(np.sum(hits_1)) / (np.sum(hits_1) +  np.sum(misses_1)),
                "Recall_5":(np.sum(hits_5)) / (np.sum(hits_5) +  np.sum(misses_5)),
                "Recall_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "Recall_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "Recall_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "Recall_50":(np.sum(hits_50)) / (np.sum(hits_50) +  np.sum(misses_50)),


                "CSI_1_m0":np.sum(hits_1_m0) / (np.sum(hits_1_m0) + np.sum(falsealarms_1_m0) + np.sum(misses_1_m0)),
                "CSI_5_m0":np.sum(hits_5_m0) / (np.sum(hits_5_m0) + np.sum(falsealarms_5_m0) + np.sum(misses_5_m0)),
                "CSI_10_m0":np.sum(hits_10_m0) / (np.sum(hits_10_m0) + np.sum(falsealarms_10_m0) + np.sum(misses_10_m0)),
                "CSI_20_m0":np.sum(hits_20_m0) / (np.sum(hits_20_m0) + np.sum(falsealarms_20_m0) + np.sum(misses_20_m0)),
                "CSI_30_m0":np.sum(hits_30_m0) / (np.sum(hits_30_m0) + np.sum(falsealarms_30_m0) + np.sum(misses_30_m0)),
                "CSI_50_m0":np.sum(hits_50_m0) / (np.sum(hits_50_m0) + np.sum(falsealarms_50_m0) + np.sum(misses_50_m0)),

                "BIAS_1_m0":(np.sum(hits_1_m0)+np.sum(falsealarms_1_m0)) / (np.sum(hits_1_m0) +  np.sum(misses_1_m0)),
                "BIAS_5_m0":(np.sum(hits_5_m0)+np.sum(falsealarms_5_m0)) / (np.sum(hits_5_m0) +  np.sum(misses_5_m0)),
                "BIAS_10_m0":(np.sum(hits_10_m0)+np.sum(falsealarms_10_m0)) / (np.sum(hits_10_m0) +  np.sum(misses_10_m0)),
                "BIAS_20_m0":(np.sum(hits_20_m0)+np.sum(falsealarms_20_m0)) / (np.sum(hits_20_m0) +  np.sum(misses_20_m0)),
                "BIAS_30_m0":(np.sum(hits_30_m0)+np.sum(falsealarms_30_m0)) / (np.sum(hits_30_m0) +  np.sum(misses_30_m0)),
                "BIAS_50_m0":(np.sum(hits_50_m0)+np.sum(falsealarms_50_m0)) / (np.sum(hits_50_m0) +  np.sum(misses_50_m0)),

                "Precision_1_m0":(np.sum(hits_1_m0)) / (np.sum(hits_1_m0) +  np.sum(falsealarms_1_m0)),
                "Precision_5_m0":(np.sum(hits_5_m0)) / (np.sum(hits_5_m0) +  np.sum(falsealarms_5_m0)),
                "Precision_10_m0":(np.sum(hits_10_m0)) / (np.sum(hits_10_m0) +  np.sum(falsealarms_10_m0)),
                "Precision_20_m0":(np.sum(hits_20_m0)) / (np.sum(hits_20_m0) +  np.sum(falsealarms_20_m0)),
                "Precision_30_m0":(np.sum(hits_30_m0)) / (np.sum(hits_30_m0) +  np.sum(falsealarms_30_m0)),
                "Precision_50_m0":(np.sum(hits_50_m0)) / (np.sum(hits_50_m0) +  np.sum(falsealarms_50_m0)),


                "Recall_1_m0":(np.sum(hits_1_m0)) / (np.sum(hits_1_m0) +  np.sum(misses_1_m0)),
                "Recall_5_m0":(np.sum(hits_5_m0)) / (np.sum(hits_5_m0) +  np.sum(misses_5_m0)),
                "Recall_10_m0":(np.sum(hits_10_m0)) / (np.sum(hits_10_m0) +  np.sum(misses_10_m0)),
                "Recall_20_m0":(np.sum(hits_20_m0)) / (np.sum(hits_20_m0) +  np.sum(misses_20_m0)),
                "Recall_30_m0":(np.sum(hits_30_m0)) / (np.sum(hits_30_m0) +  np.sum(misses_30_m0)),
                "Recall_50_m0":(np.sum(hits_50_m0)) / (np.sum(hits_50_m0) +  np.sum(misses_50_m0)),


                "CSI_1_m1":np.sum(hits_1_m1) / (np.sum(hits_1_m1) + np.sum(falsealarms_1_m1) + np.sum(misses_1_m1)),
                "CSI_5_m1":np.sum(hits_5_m1) / (np.sum(hits_5_m1) + np.sum(falsealarms_5_m1) + np.sum(misses_5_m1)),
                "CSI_10_m1":np.sum(hits_10_m1) / (np.sum(hits_10_m1) + np.sum(falsealarms_10_m1) + np.sum(misses_10_m1)),
                "CSI_20_m1":np.sum(hits_20_m1) / (np.sum(hits_20_m1) + np.sum(falsealarms_20_m1) + np.sum(misses_20_m1)),
                "CSI_30_m1":np.sum(hits_30_m1) / (np.sum(hits_30_m1) + np.sum(falsealarms_30_m1) + np.sum(misses_30_m1)),
                "CSI_50_m1":np.sum(hits_50_m1) / (np.sum(hits_50_m1) + np.sum(falsealarms_50_m1) + np.sum(misses_50_m1)),

                "BIAS_1_m1":(np.sum(hits_1_m1)+np.sum(falsealarms_1_m1)) / (np.sum(hits_1_m1) +  np.sum(misses_1_m1)),
                "BIAS_5_m1":(np.sum(hits_5_m1)+np.sum(falsealarms_5_m1)) / (np.sum(hits_5_m1) +  np.sum(misses_5_m1)),
                "BIAS_10_m1":(np.sum(hits_10_m1)+np.sum(falsealarms_10_m1)) / (np.sum(hits_10_m1) +  np.sum(misses_10_m1)),
                "BIAS_20_m1":(np.sum(hits_20_m1)+np.sum(falsealarms_20_m1)) / (np.sum(hits_20_m1) +  np.sum(misses_20_m1)),
                "BIAS_30_m1":(np.sum(hits_30_m1)+np.sum(falsealarms_30_m1)) / (np.sum(hits_30_m1) +  np.sum(misses_30_m1)),
                "BIAS_50_m1":(np.sum(hits_50_m1)+np.sum(falsealarms_50_m1)) / (np.sum(hits_50_m1) +  np.sum(misses_50_m1)),

                "Precision_1_m1":(np.sum(hits_1_m1)) / (np.sum(hits_1_m1) +  np.sum(falsealarms_1_m1)),
                "Precision_5_m1":(np.sum(hits_5_m1)) / (np.sum(hits_5_m1) +  np.sum(falsealarms_5_m1)),
                "Precision_10_m1":(np.sum(hits_10_m1)) / (np.sum(hits_10_m1) +  np.sum(falsealarms_10_m1)),
                "Precision_20_m1":(np.sum(hits_20_m1)) / (np.sum(hits_20_m1) +  np.sum(falsealarms_20_m1)),
                "Precision_30_m1":(np.sum(hits_30_m1)) / (np.sum(hits_30_m1) +  np.sum(falsealarms_30_m1)),
                "Precision_50_m1":(np.sum(hits_50_m1)) / (np.sum(hits_50_m1) +  np.sum(falsealarms_50_m1)),

                "Recall_1_m1":(np.sum(hits_1_m1)) / (np.sum(hits_1_m1) +  np.sum(misses_1_m1)),
                "Recall_5_m1":(np.sum(hits_5_m1)) / (np.sum(hits_5_m1) +  np.sum(misses_5_m1)),
                "Recall_10_m1":(np.sum(hits_10_m1)) / (np.sum(hits_10_m1) +  np.sum(misses_10_m1)),
                "Recall_20_m1":(np.sum(hits_20_m1)) / (np.sum(hits_20_m1) +  np.sum(misses_20_m1)),
                "Recall_30_m1":(np.sum(hits_30_m1)) / (np.sum(hits_30_m1) +  np.sum(misses_30_m1)),
                "Recall_50_m1":(np.sum(hits_50_m1)) / (np.sum(hits_50_m1) +  np.sum(misses_50_m1)),

                
                "CSI_1_m2":np.sum(hits_1_m2) / (np.sum(hits_1_m2) + np.sum(falsealarms_1_m2) + np.sum(misses_1_m2)),
                "CSI_5_m2":np.sum(hits_5_m2) / (np.sum(hits_5_m2) + np.sum(falsealarms_5_m2) + np.sum(misses_5_m2)),
                "CSI_10_m2":np.sum(hits_10_m2) / (np.sum(hits_10_m2) + np.sum(falsealarms_10_m2) + np.sum(misses_10_m2)),
                "CSI_20_m2":np.sum(hits_20_m2) / (np.sum(hits_20_m2) + np.sum(falsealarms_20_m2) + np.sum(misses_20_m2)),
                "CSI_30_m2":np.sum(hits_30_m2) / (np.sum(hits_30_m2) + np.sum(falsealarms_30_m2) + np.sum(misses_30_m2)),
                "CSI_50_m2":np.sum(hits_50_m2) / (np.sum(hits_50_m2) + np.sum(falsealarms_50_m2) + np.sum(misses_50_m2)),

                "BIAS_1_m2":(np.sum(hits_1_m2)+np.sum(falsealarms_1_m2)) / (np.sum(hits_1_m2) +  np.sum(misses_1_m2)),
                "BIAS_5_m2":(np.sum(hits_5_m2)+np.sum(falsealarms_5_m2)) / (np.sum(hits_5_m2) +  np.sum(misses_5_m2)),
                "BIAS_10_m2":(np.sum(hits_10_m2)+np.sum(falsealarms_10_m2)) / (np.sum(hits_10_m2) +  np.sum(misses_10_m2)),
                "BIAS_20_m2":(np.sum(hits_20_m2)+np.sum(falsealarms_20_m2)) / (np.sum(hits_20_m2) +  np.sum(misses_20_m2)),
                "BIAS_30_m2":(np.sum(hits_30_m2)+np.sum(falsealarms_30_m2)) / (np.sum(hits_30_m2) +  np.sum(misses_30_m2)),
                "BIAS_50_m2":(np.sum(hits_50_m2)+np.sum(falsealarms_50_m2)) / (np.sum(hits_50_m2) +  np.sum(misses_50_m2)),

                "Precision_1_m2":(np.sum(hits_1_m2)) / (np.sum(hits_1_m2) +  np.sum(falsealarms_1_m2)),
                "Precision_5_m2":(np.sum(hits_5_m2)) / (np.sum(hits_5_m2) +  np.sum(falsealarms_5_m2)),
                "Precision_10_m2":(np.sum(hits_10_m2)) / (np.sum(hits_10_m2) +  np.sum(falsealarms_10_m2)),
                "Precision_20_m2":(np.sum(hits_20_m2)) / (np.sum(hits_20_m2) +  np.sum(falsealarms_20_m2)),
                "Precision_30_m2":(np.sum(hits_30_m2)) / (np.sum(hits_30_m2) +  np.sum(falsealarms_30_m2)),
                "Precision_50_m2":(np.sum(hits_50_m2)) / (np.sum(hits_50_m2) +  np.sum(falsealarms_50_m2)),

                "Recall_1_m2":(np.sum(hits_1_m2)) / (np.sum(hits_1_m2) +  np.sum(misses_1_m2)),
                "Recall_5_m2":(np.sum(hits_5_m2)) / (np.sum(hits_5_m2) +  np.sum(misses_5_m2)),
                "Recall_10_m2":(np.sum(hits_10_m2)) / (np.sum(hits_10_m2) +  np.sum(misses_10_m2)),
                "Recall_20_m2":(np.sum(hits_20_m2)) / (np.sum(hits_20_m2) +  np.sum(misses_20_m2)),
                "Recall_30_m2":(np.sum(hits_30_m2)) / (np.sum(hits_30_m2) +  np.sum(misses_30_m2)),
                "Recall_50_m2":(np.sum(hits_50_m2)) / (np.sum(hits_50_m2) +  np.sum(misses_50_m2)),


                "Auc_1_area_ROC": auc_1,
                "Auc_5_area_ROC": auc_5,
                "Auc_10_area_ROC": auc_10,
                "Auc_20_area_ROC": auc_20,
                "Auc_30_area_ROC": auc_30,
                "False_predict_rate_1_ROC": fpr_1,
                "False_predict_rate_5_ROC": fpr_5,
                "False_predict_rate_10_ROC": fpr_10,
                "False_predict_rate_20_ROC": fpr_20,
                "False_predict_rate_30_ROC": fpr_30,
                "True_predict_rate_1_ROC": tpr_1,
                "True_predict_rate_5_ROC": tpr_5,
                "True_predict_rate_10_ROC": tpr_10,
                "True_predict_rate_20_ROC": tpr_20,
                "True_predict_rate_30_ROC": tpr_30,
                "Thresholds_1_ROC": thresholds_1,
                "Thresholds_5_ROC": thresholds_5,
                "Thresholds_10_ROC": thresholds_10,
                "Thresholds_20_ROC": thresholds_20,
                "Thresholds_30_ROC": thresholds_30,

                "Prob_true_SK_reliability_1":prob_true_1,
                "Prob_true_SK_reliability_5":prob_true_5,
                "Prob_true_SK_reliability_10":prob_true_10,
                "Prob_true_SK_reliability_20":prob_true_20,
                "Prob_true_SK_reliability_30":prob_true_30,

                "Prob_pred_SK_reliability_1":prob_pred_1,
                "Prob_pred_SK_reliability_5":prob_pred_5,
                "Prob_pred_SK_reliability_10":prob_pred_10,
                "Prob_pred_SK_reliability_20":prob_pred_20,
                "Prob_pred_SK_reliability_30":prob_pred_30,

              }
    
    
    return metricses

####end CGAN-all scores



def par_gen_patch_eval_junxu(gen, dl_test, nens, ds_min, ds_max, tp_log, device):    ##cgan评分计算
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    rhist = []
    rels_20 = []
    rels_30 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_10 = []
    preds_brier_20 = []
    preds_brier_30 = []
    hits_10 = []
    misses_10 = []
    falsealarms_10 = []
    correctnegatives_10 = []
    hits_20 = []
    misses_20 =[]
    falsealarms_20 = []
    correctnegatives_20 = []
    hits_30 = []
    misses_30 = []
    falsealarms_30 = []
    correctnegatives_30 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            rmse.append(res[1])
            rhist.append(res[2])
            rels_20.append(res[3])
            rels_30.append(res[4])
            preds_brier_10.append(res[5])
            preds_brier_20.append(res[6])
            preds_brier_30.append(res[7])
            hits_10.append(res[8])
            misses_10.append(res[9])
            falsealarms_10.append(res[10])
            correctnegatives_10.append(res[11])
            hits_20.append(res[12])
            misses_20.append(res[13])
            falsealarms_20.append(res[14])
            correctnegatives_20.append(res[15])
            hits_30.append(res[16])
            misses_30.append(res[17])
            falsealarms_30.append(res[18])
            correctnegatives_30.append(res[19])


            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y,href) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 20, window=20, device=device) ###member平均的FSS
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0))) ###0member的preds
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0]) ###0member的hist
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape) ##扰动
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) ##扰动

        pool.starmap_async(compute_metrics_junxu, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_20 = xr.concat(rels_20, dim = "patch")
    weights_20 = rels_20.samples / rels_20.samples.sum(dim="patch")
    weighted_relative_freq_20 = (weights_20*rels_20.relative_freq).sum(dim="patch")
    samples_20 = rels_20.samples.sum(dim="patch")
    forecast_probs_20 = rels_20.forecast_probability
    
    rels_30 = xr.concat(rels_30, dim = "patch")
    weights_30 = rels_30.samples / rels_30.samples.sum(dim="patch")
    weighted_relative_freq_30 = (weights_30*rels_30.relative_freq).sum(dim="patch")
    samples_30 = rels_30.samples.sum(dim="patch")
    forecast_probs_30 = rels_30.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "rankhist": rhist, 
               "reliability_20": (weighted_relative_freq_20, forecast_probs_20, samples_20), 
               "reliability_30": (weighted_relative_freq_30, forecast_probs_30, samples_30), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_10" : np.mean(preds_brier_10),
                "preds_brier_20" : np.mean(preds_brier_20),
                "preds_brier_30": np.mean(preds_brier_30),
                "CSI_10":np.sum(hits_10) / (np.sum(hits_10) + np.sum(falsealarms_10) + np.sum(misses_10)),
                "CSI_20":np.sum(hits_20) / (np.sum(hits_20) + np.sum(falsealarms_20) + np.sum(misses_20)),
                "CSI_30":np.sum(hits_30) / (np.sum(hits_30) + np.sum(falsealarms_30) + np.sum(misses_30)),
                "BIAS_10":(np.sum(hits_10)+np.sum(falsealarms_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "BIAS_20":(np.sum(hits_20)+np.sum(falsealarms_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "BIAS_30":(np.sum(hits_30)+np.sum(falsealarms_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
                "Precision_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(falsealarms_10)),
                "Precision_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(falsealarms_20)),
                "Precision_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(falsealarms_30)),
                "Recall_10":(np.sum(hits_10)) / (np.sum(hits_10) +  np.sum(misses_10)),
                "Recall_20":(np.sum(hits_20)) / (np.sum(hits_20) +  np.sum(misses_20)),
                "Recall_30":(np.sum(hits_30)) / (np.sum(hits_30) +  np.sum(misses_30)),
              }
    
    
    return metrics


def fss_single(x,y,threshold, window, device):
    x_mask = x>=threshold
    y_mask = y>=threshold
    window_size=window**2
#    mse = []
    yin = torch.from_numpy(y_mask.values.astype(np.float32)).unsqueeze(1).to(device)
    print('this is truth shape')
    print(yin.shape)
    y_out = F.avg_pool2d(yin, window, stride=1, padding=0)
    print('this is truthout shape')
    print(y_out.shape)
    
    xin = torch.from_numpy(x_mask[:,:,:].values.astype(np.float32)).to(device)
    print('this is pred shape')
    print(xin.shape)
    x_out = F.avg_pool2d(xin, window, stride=1, padding=0)
    print('this is predout shape')
    print(x_out.shape)
    mseij = torch.mean(torch.square(x_out - y_out))
    print('this is fenzi')
    print(mseij)    
    mse_ref = torch.mean(torch.square(x_out)) +  torch.mean(torch.square(y_out))
    if mse_ref == 0:
     print('fenmu 0:')
     #mse=np.zeros(1)
     #mse=torch.from_numpy(mse)
     #print(mse)
     mse=np.nan
    else:
     fss_ij = 1 - (mseij / mse_ref)
     mse=fss_ij
     print(mse)
#    mse.append(fss_ij.detach().cpu().numpy())
             
    return mse.detach().cpu().numpy()

def href_patch_eval(dl_test, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
#         print(x.shape)
#         preds = x.squeeze()
#         print(preds.shape)
        preds = x.numpy().transpose(1,0,2,3)
        truth = y.numpy().squeeze(1)
        
#         print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min


        if tp_log:
            truth = log_retrans(truth, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(11)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics


def tigge_interp_patch_eval(dl_test, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        print(x.shape)
        preds = F.interpolate(x, size = (128, 128), mode='bilinear').detach().to('cpu').numpy().squeeze()
        print(preds.shape)
        preds = preds.transpose(1,0,2,3)
        truth = y.numpy().squeeze(1)
        
        print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(11)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics


def compute_metrics(truth, preds, truth_pert, preds_pert, sample):
    sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
    truth_course = truth.coarsen(lat=4, lon=4)
    preds_course = preds.coarsen(lat=4, lon=4)
    sample_max_pool_crps = xs.crps_ensemble(truth_course.max().sel(sample=sample), preds_course.max().sel(sample=sample)).values
    sample_avg_pool_crps = xs.crps_ensemble(truth_course.mean().sel(sample=sample), preds_course.mean().sel(sample=sample)).values
    sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
    rhist = xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values
    
    rel1 = xs.reliability(truth.sel(sample=sample)>1,(preds.sel(sample=sample)>1).mean('member'))
    rel1 = xr.where(np.isnan(rel1), 0, rel1)
    rel1['relative_freq'] = rel1
    
    rel4 = xs.reliability(truth.sel(sample=sample)>4,(preds.sel(sample=sample)>4).mean('member'))
    rel4 = xr.where(np.isnan(rel4), 0, rel4)
    rel4['relative_freq'] = rel4

    
    sample_brier_1 = xs.brier_score(truth.sel(sample=sample) > 1.0, (preds.sel(sample=sample) > 1.0).mean('member'), dim=['lat', 'lon'])
    
    sample_brier_5 = xs.brier_score(truth.sel(sample=sample) > 5.0, (preds.sel(sample=sample) > 5.0).mean('member'), dim=['lat', 'lon'])
        
    sample_brier_10 = xs.brier_score(truth.sel(sample=sample) > 10.0, (preds.sel(sample=sample) > 10.0).mean('member'), dim=['lat', 'lon'])
    
    return (sample_crps, sample_max_pool_crps, sample_avg_pool_crps, sample_rmse, rhist, rel1, rel4, sample_brier_1, sample_brier_5, sample_brier_10)
    

    
def fss(x,y,threshold, window, device):
    x_mask = x>=threshold
    y_mask = y>=threshold
    window_size=window**2
    mse = []
    yin = torch.from_numpy(y_mask.values.astype(np.float32)).unsqueeze(1).to(device)
    y_out = F.avg_pool2d(yin, window, stride=1, padding=0)
    for member in range(x_mask.shape[1]):
        xin = torch.from_numpy(x_mask[:,member:member+1,:,:].values.astype(np.float32)).to(device)
        x_out = F.avg_pool2d(xin, window, stride=1, padding=0)
        mseij = torch.mean(torch.square(x_out - y_out))    
        mse_ref = torch.mean(torch.square(x_out)) +  torch.mean(torch.square(y_out))
        if mse_ref == 0:
            continue
        fss_ij = 1 - (mseij / mse_ref)
        mse.append(fss_ij.detach().cpu().numpy())
        #print(mse)
             
    return np.mean(mse)
       
    #for sample in range(x_mask.shape[0])for sample in range(x_mask.shape[0]):
#        yin = torch.from_numpy(y_mask[sample:sample+1,:,:].values.astype(np.float32)).unsqueeze(1).to(device)
#        y_out = conv(yin)/window_size
#        for member in range(x_mask.shape[1]):
#            xin = torch.from_numpy(x_mask[sample:sample+1,member:member+1,:,:].values.astype(np.float32)).transpose(0,1).to(device)  
#            x_out = conv(xin)/window_size
#            mseij = torch.mean(torch.square(x_out - y_out))    
#            mse_ref = torch.mean(torch.square(x_out)) +  torch.mean(torch.square(y_out))
#            if mse_ref == 0:
#                continue
#            fss_ij = 1 - (mseij / mse_ref)
#            mse.append(fss_ij.detach().cpu().numpy())
            
#    return np.mean(mse)

def par_gen_full_field_eval(gen, ds_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """  
    
    timer = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    
    timer.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    timer.toc('Setting up the pool took')
    
    print(f"Total batches: {len(ds_test.tigge.valid_time)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    full_preds = []
    full_truth = []
    for idx, t in enumerate(tqdm(range(len(ds_test.tigge.valid_time)))):
        x, y = ds_test.return_full_array(t)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)

        full_preds.append(preds)
        truth = y.squeeze(0)
        full_truth.append(truth)
        if idx>30:
            break

    timer.tic()
    
    preds = np.array(full_preds)
    truth = np.array(full_truth)
    

    preds = xr.DataArray(
                preds,
                dims=['sample', 'member', 'lat', 'lon'],
                name='tp'
            )
        
    truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )

    truth = truth * (ds_max - ds_min) + ds_min

    preds = preds * (ds_max - ds_min) + ds_min

    if tp_log:
        truth = log_retrans(truth, tp_log)
        preds = log_retrans(preds, tp_log)
        
    timer.toc("xarray, scaling done in", restart=True)
    # get mask   
    ds = xr.open_dataset('/home/jupyter/data/hrrr/raw/total_precipitation/20180215_00.nc')
    ds_regridded = regrid(ds, 4, lons=(235, 290), lats=(50, 20))
    hrrr_mask = np.isfinite(ds_regridded).tp.isel(init_time=0, lead_time=0)
    
    timer.toc("hrrr mask loaded in", restart=True)
    rq = xr.open_dataarray(f'/home/jupyter/data/mrms/4km/RadarQuality.nc')
    mrms_mask = rq>-1
    mrms_mask = mrms_mask.assign_coords({
        'lat': hrrr_mask.lat,
        'lon': hrrr_mask.lon
    })
    total_mask = mrms_mask * hrrr_mask
    total_mask = total_mask.isel(lat=slice(0, -6))
    total_mask = total_mask.assign_coords({'lat': truth.lat.values, 'lon': truth.lon.values})
    
    timer.toc("total mask computed in", restart=True)
#     apply mask
    truth = truth.where(total_mask)
    preds = preds.where(total_mask)
    
    timer.toc("mask applied in", restart=True)
    # compute fss
    mean_fss = fss(preds,truth, 4, 25, device)
    
    eps = 1e-6
    bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
    pred_means.append(np.mean(preds.sel(member=0)))
    pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
    truth_means.append(np.mean(truth))
    truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])

    truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
    preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

    pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(preds.shape[0])], callback=log_result).wait()

        
        
    timer.toc("compute metrics completed in", restart=True)
    
    rels_1 = xr.concat(rels_1, dim = "time")
    print(rels_1)
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="time")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="time")
    samples_1 = rels_1.samples.sum(dim="time")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "time")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="time")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="time")
    samples_4 = rels_4.samples.sum(dim="time")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": mean_fss
              }
    
    
    return metrics

def par_SR_gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(x.shape[1]):
            noise = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x[:,i:i+1,:,:], noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics
    
def par_gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y,href) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics




def gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = xr.DataArray(data = np.zeros(nens+1), dims = "rank")
    rels = []
    for batch_idx, (x,y) in enumerate(dl_test):
        print(f"batch {batch_idx} out of {len(dl_test)}")
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min
    
        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape)

        
        t.tic()
        for sample in range(x.shape[0]):

            sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
            truth_course = truth.coarsen(lat=4, lon=4)
            preds_course = preds.coarsen(lat=4, lon=4)
            sample_max_pool_crps = xs.crps_ensemble(truth_course.max().sel(sample=sample), preds_course.max().sel(sample=sample)).values
            sample_avg_pool_crps = xs.crps_ensemble(truth_course.mean().sel(sample=sample), preds_course.mean().sel(sample=sample)).values
            crps.append(sample_crps)
            max_pool_crps.append(sample_max_pool_crps)
            avg_pool_crps.append(sample_avg_pool_crps)
            
#             t.toc('crps took', restart=True)
            
            sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
            rmse.append(sample_rmse)
            
#             t.tic()
            rhist += xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values
#             t.toc('rank histogram took', restart=True)
            
#             t.tic()
            rel = xs.reliability(truth.sel(sample=sample)>1,(preds.sel(sample=sample)>1).mean('member'))
            rel = xr.where(np.isnan(rel), 0, rel)
            rel['relative_freq'] = rel
            rels.append(rel)
#             t.toc('reliability took', restart=True)
            
        t.toc('metrics took', restart=True)

        
    rels = xr.concat(rels, dim = "patch")
    weights = rels.samples / rels.samples.sum(dim="patch")
    weighted_relative_freq = (weights*rels.relative_freq).sum(dim="patch")
    samples = rels.samples.sum(dim="patch")
    forecast_probs = rels.forecast_probability
    
    return np.mean(crps), np.mean(max_pool_crps), np.mean(avg_pool_crps), rhist, (weighted_relative_freq, forecast_probs, samples), np.mean(rmse)

def single_full_test_prediction(gen, ds_test, device):
    # Get predictions for full field
    
    preds = []
    for t in tqdm.tqdm(range(len(ds_test.tigge.valid_time))):
        x, y = ds_test.return_full_array(t)
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
        pred = gen(x, noise).to('cpu').detach().numpy().squeeze()
        preds.append(pred)
    preds = np.array(preds)
    
    # Unscale
    preds = preds * (ds_test.maxs.tp.values - ds_test.mins.tp.values) + ds_test.mins.tp.values
    
    # Un-log
    if ds_test.tp_log:
        preds = log_retrans(preds, ds_test.tp_log)
    
    # Convert to xarray
    preds = xr.DataArray(
        preds,
        dims=['valid_time', 'lat', 'lon'],
        coords={
            'valid_time': ds_test.tigge.valid_time,
            'lat': ds_test.mrms.lat.isel(
                lat=slice(ds_test.pad_mrms, ds_test.pad_mrms+preds.shape[1])
            ),
            'lon': ds_test.mrms.lon.isel(
                lon=slice(ds_test.pad_mrms, ds_test.pad_mrms+preds.shape[2])
            )
        },
        name='tp'
    )
    return preds
    
    
def ensemble_full_test_predictions(gen, ds_test, nens, device):
    """Wrapper to create ensemble"""
    preds = [single_full_test_prediction(gen, ds_test, device) for _ in range(nens)]
    return xr.concat(preds, 'member')   

def make_eval_mask():
    # mask ground truth data
    rq = xr.open_dataarray('/datadrive_ssd/mrms/4km/RadarQuality.nc')
    eval_mask = rq>-1
    fn = "/datadrive_ssd/mrms/4km/RadarOnly_QPE_06H/RadarOnly_QPE_06H_00.00_20180101-000000.nc"
    ds = xr.open_dataset(fn)
    assert eval_mask.lat.shape ==ds.lat.shape
    eval_mask['lat'] = ds.lat 
    assert eval_mask.lon.shape ==ds.lon.shape
    eval_mask['lon'] = ds.lon
    return eval_mask

def get_full_masked_mrms(gen, ds_test, device):
    #get pred to align lat and lon
    preds = single_full_test_prediction(gen, ds_test,  device);

    eval_mask = make_eval_mask()
    
    mrms = ds_test.mrms.sel(lat=preds.lat.values, 
                            lon=preds.lon.values).rename(
                            {'time': 'valid_time'}) * ds_test.maxs.tp.values
    if ds_test.tp_log:
        mrms = log_retrans(mrms, ds_test.tp_log)
    
    mrms = mrms.where(eval_mask)
    
    return mrms

def gen_full_eval(gen, ds_test, mrms, nens, device):
    preds = ensemble_full_test_predictions(gen, ds_test, nens, device)  
#     print(preds)
    crps = xs.crps_ensemble(mrms, preds).values
    rmse = xs.rmse(preds.mean('member'), mrms, dim=['lat', 'lon', 'valid_time'], skipna=True).values
    
    return preds, crps, rmse


def interpolation_full_baseline(ds_test, mrms):
    tigge = ds_test.tigge.isel(variable=0) * ds_test.maxs.tp.values
    tigge = log_retrans(tigge, ds_test.tp_log)
    #interpolate
    interp = tigge.interp_like(mrms, method='linear')   
    #calculate error
    rmse = xs.rmse(interp, mrms, dim=['lat', 'lon', 'valid_time'], skipna=True).values
    
    return tigge, interp, rmse

""" Evaluation functions and classes (??) 
# Let's ignore ensemble dimension for a start. 
# TODO for later: Add ensemble option

General workflow structure: 
1. Load in the evaluation data. Do we want to use the dataloader class? Yes! 
    Here, we need the tigge data also for different lead times as well as the different
    ensembles (ignor for now). 
    Leadtimes are already included as option in the dataloader, 
    also the minmax and tp-log scaling can be switched off per options. 

2. Compute baseline or baselines, if several are considered. 

3. Compute metrics. 
    a) Deterministic: FSS, RMSE, F1 
    b) Ensemble: CRPS, Rank histograms
    c) Realism: precip amount histogram/spectra, cell size distributions
"""

#------------ FSS-functions ( copied and adapted from L. Scheck.) ----------------
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . S C O R E _ F S S
#  compute Fractions (skill) score and related quantities (copied from L. Scheck)
#
#  Almost completely adapted from
#  Faggian, Roux, Steinle, Ebert (2015) "Fast calculation of the fractions skill score"
#  MAUSAM, 66, 3, 457-466
#
#"""
#.. module:: score_fss
#:platform: Unix
#:synopsis: Compute the fraction skill score (2D).
#.. moduleauthor:: Nathan Faggian <n.faggian@bom.gov.au>
#"""

def _compute_integral_table(field) :
    return field.cumsum(1).cumsum(0)


def _integral_filter(field, n, table=None) :
    """
    Fast summed area table version of the sliding accumulator.
    :param field: nd-array of binary hits/misses.
    :param n: window size.
    """
    w = n // 2
    if w < 1. :
        return field
    if table is None:
        table = _compute_integral_table(field)

    r, c = np.mgrid[ 0:field.shape[0], 0:field.shape[1] ]
    r = r.astype(np.int)
    c = c.astype(np.int)
    w = np.int(w)
    r0, c0 = (np.clip(r - w, 0, field.shape[0] - 1), np.clip(c - w, 0, field.shape[1] - 1))
    r1, c1 = (np.clip(r + w, 0, field.shape[0] - 1), np.clip(c + w, 0, field.shape[1] - 1))
    integral_table = np.zeros(field.shape).astype(np.int64)
    integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape))
    integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape))
    return integral_table


def _fss(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None):
    """
    Compute the fraction skill score using summed area tables .
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = _integral_filter( fcst > threshold, window, fcst_cache )
    ohat = _integral_filter( obs  > threshold, window, obs_cache  )

    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def _fss_frame(fcst, obs, windows, levels):
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: list, dataframes of the FSS: numerator,denominator and score.
    """
    num_data, den_data, fss_data = [], [], []
    #print(fcst.shape)
    #print(obs.shape)
    for level in levels:
        ftable = _compute_integral_table( fcst > level )
        otable = _compute_integral_table( obs  > level )
        _data = [_fss(fcst, obs, level, w, ftable, otable) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])
    return np.array(fss_data) #pd.DataFrame(fss_data, index=levels, columns=windows)
# ------------------- Done with FSS functions -----------------------

def _my_f1_score(obs,fcst, thresholds, **kws):
    """ wraps scikit-learn f1-score computation to fit the needs of the apply-ufunc seting.
    obs: 2d np.array of observation data 
    fcst: 2d np.array of forecast data 
    thresholds: list of precipitation thresholds to apply. Computes f1-score for each threshold. 

    Returns np.array of f1-scores, with each value belonging to a different threshold
    
    """ 
    assert obs.shape==fcst.shape,'Shapes of obs and fcst do not match.'
    f1_scores = [f1_score((obs>threshold).ravel(), (fcst>threshold).ravel(),**kws) for threshold in thresholds]
    
    return np.array(f1_scores)



def compare_fields(X, y_G = None, y_b = None, y=None, 
                   levels = np.arange(0,10,0.1), cmap='viridis', 
                   eval_mask = None ): 
    """ Compare precipitation fields by making example plots
    TODO: make map plots
    TODO: include eval_mask to shade invalid data
    X: TIGGE field 
    y_G: downscaled field from Generator/CNN, ... 
    y_b: baseline 
    y: radar precip
    """
    
    fig, axs = plt.subplots(1, 4, figsize=[20,4], sharey=True, squeeze=True)
    settings = dict(levels=levels, cmap=cmap)

    X.plot(ax=axs[0], **settings)
    axs[0].set_title('Tigge original')
    
    #if y_G:
    y_G.plot(ax=axs[1], **settings)
    axs[1].set_title('downscaled')
    
    y_b.plot(ax=axs[2], **settings)
    axs[2].set_title('baseline')
    
    y.plot(ax=axs[3], **settings)
    axs[3].set_title('MRMS')
    
    return fig, axs

def get_hrrr_mask(year = '2020', fdir = '/datadrive/hrrr/4km/'):
    """ Function to get the hrrr mask from interpolation. 
    Not very elegantly coded: we compute the yearly maximum
    at each grid point. Assuming that every valid grid point rains, 
    the mask is defined to exclude all grid points with zero precip 
    in the whole year. 
    year: year as str to be used 
    fdir: directory where to save the resulting data
    """
    try: 
        fn_mask = fdir+year+'_hrrr_interpolation_mask.nc'
        return xr.open_dataarray(fn_mask) 
    except:
        fn = '/datadrive/hrrr/4km/total_precipitation/'+year+'*.nc'
        hrrr_ds = xr.open_mfdataset(fn)
        hrrr_ds= hrrr_ds.tp.diff('lead_time').sel(lead_time =np.timedelta64(12, 'h'))
        hrrr_ds['valid_time'] = hrrr_ds.init_time + hrrr_ds.lead_time
        hrrr_ds= hrrr_ds.swap_dims({'init_time': 'valid_time'})
        mask = (hrrr_ds.max('valid_time')>0)
        
        process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        git_head_hash = process.communicate()[0].strip()
        mask.attrs['git_hash_at_creation'] = str(git_head_hash)
        mask.attrs['year']=year
        mask.attrs['date_of_computation'] = date.today().strftime("%d/%m/%Y")

        # I don't have writing permission
        #mask.to_netcdf(fdir+year+'_hrrr_interpolation_mask.nc')
        return mask.compute() 

def get_eval_mask(criterion='radarquality', rq_threshold = -1, 
                rq_fn = '/datadrive/mrms/4km/RadarQuality.nc', ds = None): 
    """ Returns a lon-lat mask which area we evaluate on. 
        The radar quality mask is used to determine this. 

        criterion: criterion to apply. ('radarquality', 'hrr+radar) 
        rq_threshold: threshold for 'radarquality'-criterion.   
                    -1 covers everything with radar availability. 
                    Larger thresholds require higher quality. 
        ds: If patchareas criterion, the tiggemrmsm object is required 
                    to get the radarmask for the patches.

        Returns: boolean xr-dataarray, with same lon-lat dimensions 
            as the radar data. 
    """
    # TODO: consider time dependece of radarmask! (do we need to inlcude that?)
    if criterion in ['radarquality', 'hrr+radar']: # use rq>rq-threshold as criterion 
        rq = xr.open_dataarray(rq_fn)
        eval_mask = rq>rq_threshold

        # hardcode: get proper lon-lat values for rq-mask. Somehow weird!
        fn = "/datadrive/mrms_old/4km/RadarOnly_QPE_06H/MRMS_RadarOnly_QPE_06H_00.00_20201001-000000.nc"
        ds = xr.open_dataset(fn)
        assert eval_mask.lat.shape ==ds.lat.shape
        eval_mask['lat'] = ds.lat 
        assert eval_mask.lon.shape ==ds.lon.shape
        eval_mask['lon'] = ds.lon
    
    if criterion =='hrr+radar': 
        hrrr_mask = get_hrrr_mask(year = '2020', fdir = '/datadrive/')
        # somehow lon lats are not absolutely identical! 
        assert hrrr_mask.lat.shape ==ds.lat.shape
        hrrr_mask['lat'] = ds.lat 
        assert hrrr_mask.lon.shape ==ds.lon.shape
        hrrr_mask['lon'] = ds.lon

        eval_mask = eval_mask * hrrr_mask 

    return eval_mask


def get_baseline(X, y, kind = 'interpol', 
                 X_lon=None, X_lat=None, y_lon = None, y_lat =None, 
                 HRRR_fdir = '/datadrive/hrrr/4km/total_precipitation/' ): 
    """ Function computes baseline, i.e. interpolates X onto the grid of y. 
    
    This is probably overkill for now, but might be handy later on when we have different baselines

    1. If X and y are given as numpy arrays, transform to xarray 
    2. Apply interpolation
    
    X: Tigge dataset, or sample. Can be xarray format or numpy
    y: Target radar dataset or radar sample corresponding to X. Can be xarray format or numpy 
    kind: kind of baseline to use: ['interpol', 'HRRR']
    X_lon,X_lat: arrays of longitudes and latitudes for X. Need to be specified only if X is numpy array.
    y_lon,y_lat: arrays of longitudes and latitudes for y. Need to be specified only if y is numpy array.
    
    Returns: baseline downscaled X to the grid of y
    """
    # make sure we have xarrays as inputs, makes interpolation easier
    if type(X) != xr.DataArray:
        X = xr.DataArray(data=X, dims=["lat", "lon"], 
                         coords=dict(lon=("lon", X_lon), lat=("lat", X_lat)))
        # iterative function call:
        return get_baseline(X,y,kind=kind,**kws) 
    if type(y) != xr.DataArray:
        y = xr.DataArray(data=y, dims=["lat", "lon"], 
                         coords=dict(lon=("lon", y_lon), lat=("lat", y_lat)))
        # iterative function call:
        return get_baseline(X,y,kind = kind, **kws)
    
    assert type(X) == xr.DataArray, 'X is not an xarray.'
    assert type(y) == xr.DataArray, 'y is not an xarray.'
    #assert X.dims == y.dims, 'Dimensions of X and y do not match.'
    
        
    # Do the interpolation 
    if kind =='interpol': 
        y_baseline = X.interp_like(y, kwargs = dict(fill_value='extrapolate')) 
        # fill_value: for scipy interoplate, extrapolates values at the boundaries, so no Nans appear! 
    elif kind =='HRRR': # load HRRR data 
        hrrr = xr.open_mfdataset(HRRR_fdir+ '*')
        hrrr= hrrr.tp.diff('lead_time').sel(lead_time = X.lead_time)
        hrrr['valid_time'] = hrrr.init_time + hrrr.lead_time
        hrrr= hrrr.swap_dims({'init_time': 'valid_time'})
        y_baseline = hrrr # This is not tested and probably not yet finished!
        
    assert y_baseline.shape == y.shape, 'y_baseline and y do not have the same size!'
    return y_baseline
    
def compute_eval_metrics(fcst, obs, eval_mask = None, metrics = ['RMSE', 'FSS', 'F1'],
                     fss_scales=[41,61], fss_thresholds = [1., 5.],
                     f1_thresholds = [0.1,1., 5.], f1_kws=dict() ): 
    """ Function to compute evaluation metrics to compare a forecast with observations
    fcst: xr-array of the forecast, e.g. the interpolation baseline or the downscaled forecast
    obs: xr-array fo observations, i.e. radar data 
    eval_mask: 2d boolean mask to apply the evaluation on, i.e. radar quality mask
    metrics: list of metrics to consider
    
    fss_scales: list of spatial scales to use for the FSS calculation
    fss_thresholds: list of precip thresholds to use for the FSS calculation
    
    Returns: xr-dataset with different metrics as different variables   
        Note that this function utilizes dask and returns xarrays with dask arrays. Use
    """
    
    
        
    # rechunking necessary for performance 
    fcst = fcst.chunk({'valid_time':1})
    obs = obs.chunk({'valid_time':1}) 
    
    # apply eval mask: 
    if eval_mask is not None:
        fcst = fcst.where(eval_mask)
        obs = obs.where(eval_mask)
        
    metrics_ds = xr.Dataset()
    # RMSE 
    if 'RMSE' in metrics: 
        rmse = xs.rmse(fcst, 
            obs, dim=['lon', 'lat'], skipna=True)
        metrics_ds['RMSE'] = rmse
        
    if 'FSS' in metrics: # maybe there is a way to implement this faster?
        from dask.diagnostics import ProgressBar

        fss_da = xr.apply_ufunc(_fss_frame, fcst, obs, input_core_dims=[[ 'lat', 'lon'], ['lat', 'lon']],
                       output_core_dims=[['fss_thresholds','fss_scales']], 
                       output_dtypes=[fcst.dtype],
                       dask_gufunc_kwargs = dict(output_sizes= {'fss_scales':len(fss_scales), 'fss_thresholds': len(fss_thresholds)},),
                       vectorize =True, dask='parallelized',
                       kwargs = dict(windows=fss_scales, levels = fss_thresholds))
        fss_da['fss_thresholds'] = np.array(fss_thresholds)
        fss_da['fss_scales'] = np.array(fss_scales)
        #with ProgressBar(minimum=1): 
            #fss_da = fss_da.compute()
        metrics_ds['FSS'] = fss_da

    if 'F1' in metrics: 
        f1_da = xr.apply_ufunc(_my_f1_score, (obs.where(eval_mask)), (fcst.where(eval_mask)), 
                    input_core_dims=[['lat', 'lon'], ['lat', 'lon']], output_dtypes=[fcst.dtype],
                    output_core_dims=[['f1_thresholds']], vectorize=True, dask='parallelized',
                    dask_gufunc_kwargs = dict(output_sizes= {'f1_thresholds': len(f1_thresholds)},),
                    kwargs = dict(thresholds=f1_thresholds, **f1_kws))
        f1_da['f1_thresholds'] = f1_thresholds
        f1_da.name = 'F1-Score'
        metrics_ds['F1-Score'] = f1_da
    
    return metrics_ds


def evaluate_downscaled_fcst(coarse_fcst, downscaled_fcst, obs, baselines = ['interpol', 'HRRR'],  save_to=None, **kws):
    """ xarrays as input 
    Parameter: 
    coarse_fcst (e.g. tigge data): xr.dataarray precipitation
    downscaled_fcst (e.g. gan-generated): xr.dataarray precipitation
    obs (e.g. radar): xr.dataarray precipiatation
    save_to: string, if given, saves eval-metrics to file as specified by string
    """

    # Step 1: data preparation: matching time dimensions, matching lon-lat dimensions
    if type(downscaled_fcst) is not xr.DataArray: # select variable "tp"
        try: downscaled_fcst=downscaled_fcst.tp 
        except: "downscaled_fcst input must be a xr.dataarray." 
    if 'variable' in coarse_fcst.coords: # select tp from coord "variable"
        try: coarse_fcst=coarse_fcst.sel(variable='tp')
        except: "coars_fcst input must be precipitation only." 
        
    obs = obs.sel(lat=downscaled_fcst.lat, lon=downscaled_fcst.lon)
    obs = obs.sel(time = downscaled_fcst.valid_time.values)    
    obs = obs.rename({'time':'valid_time'})  # we need consistent time naming
        
    
        
    # Step 2: compute baselines
    bl_dict = dict()
    for bl in baselines: 
        bl_dict[bl] = get_baseline(coarse_fcst, obs, kind=bl)
    #assert baseline_0.shape == downscaled_fcst.shape, 'baseline and downscaled_fcst have different shapes!'
        

    
    # Step 3: evaluation mask
    eval_mask = get_eval_mask()
    eval_mask = eval_mask.sel(lat=downscaled_fcst.lat, lon=downscaled_fcst.lon)

    # Step 4: compute different metrics
    metrics_list = []
    for bl, fcst in bl_dict.items():
        metrics = compute_eval_metrics(fcst, obs, eval_mask )       
        metrics['fcst_type']  = bl + '_baseline'
        metrics_list.append(metrics)
    
    metrics_dfcst = compute_eval_metrics(downscaled_fcst, obs, eval_mask, **kws)
    metrics_dfcst['fcst_type'] = 'Generator'
    metrics_list.append(metrics_dfcst)
    metrics = xr.concat(metrics_list,dim = "fcst_type")
    metrics
    
    print("Compute metrics:")
    with ProgressBar():
        metrics.load() # execute fss computation  
    
    # Step 5: save metrics to file 
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    git_head_hash
    metrics.attrs['git_hash_at_creation'] = git_head_hash
    metrics.attrs['date_of_computation'] = date.today().strftime("%d/%m/%Y")
    if save_to: 
        metrics.to_netcdf(save_to)

    return metrics
    



def _main(lead_time = 12): 
    
    # 1. Load in data: 
    ds = TiggeMRMSDataset(
    tigge_dir='/datadrive/tigge/32km/',
    tigge_vars=['total_precipitation'],
    mrms_dir='/datadrive/mrms/4km/RadarOnly_QPE_06H/',
    rq_fn='/datadrive/mrms/4km/RadarQuality.nc',
    val_days=7,
    split='valid',
    tp_log=0, scale=False,
    lead_time=12, # Sofar, this can not yet handle arrays of lead time. 
    ) 


    # 2. Compute baseline 
    tigge = ds.tigge.isel(variable=0)
    mrms = ds.mrms
    baseline = get_baseline(tigge, mrms)


    # 3. evaluation mask
    eval_mask = get_eval_mask()
    eval_mask['lat']=mrms.lat # somehow rq has weird lon lat values! 
    eval_mask['lon']=mrms.lon

    # 4. compute metrics: 
    metrics = compute_eval_metrics(baseline, mrms, eval_mask )
    with ProgressBar():
        metrics.load() # execute fss computation    


if __name__ == '__main__':
    Fire(_main)
