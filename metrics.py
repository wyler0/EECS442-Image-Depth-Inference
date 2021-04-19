"""
EECS 442 - Final Proj

Implementation of 5 evaluation meetrics:
ARE, ALE, RMSE, MAE, thresh_acc
"""

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def ARE(gt_depth_map, pred_depth_map):
    # gt_depth_map bs x h x w x 1
    # pred_depth_map bs x h x w x 1
    error = ((gt_depth_map - pred_depth_map).abs()/gt_depth_map).mean()
    return error.item()

def ALE(gt_depth_map, pred_depth_map):
    # gt_depth_map bs x h x w x 1
    # pred_depth_map bs x h x w x 1
    gt_depth_map_log = torch.log10(gt_depth_map)
    # gt_depth_map_log[torch.isnan(gt_depth_map_log)] = 0
    pred_depth_map_log = torch.log10(pred_depth_map)
    # pred_depth_map_log[torch.isnan(pred_depth_map_log)] = 0
    error = (gt_depth_map_log - pred_depth_map_log).abs().mean()
    return error.item()

def RMSE(gt_depth_map, pred_depth_map):
    # gt_depth_map bs x h x w x 1
    # pred_depth_map bs x h x w x 1
    error = (gt_depth_map - pred_depth_map).pow(2).sum(-1).sum(-1).sum(-1).sqrt()
    error = error.mean()
    return error.item()

def MAE(gt_depth_map, pred_depth_map):
    # gt_depth_map bs x h x w x 1
    # pred_depth_map bs x h x w x 1
    error = (gt_depth_map - pred_depth_map).abs().mean() #sum(-1).sum(-1).sum(-1)
    # error = error.mean()
    return error.item()

def thresh_acc(gt_depth_map, pred_depth_map, threshold=1.25):
    return 0; # TEMPORARY due to GPU mem limits
    bs, h, w, _ = gt_depth_map.size()
    index = torch.where(torch.max(gt_depth_map/pred_depth_map, pred_depth_map/gt_depth_map) < threshold)
    percent = len(index[0])/(bs*h*w)
    return percent

def plot_metrics(train_metrics, val_metrics, test_metrics):
    return
    # plot metrics across dataset splits
    plt.subplot(2, 3, 1)
    data = {'train_MSE': train_metrics[0], 'val_MSE': val_metrics[0], 'test_MSE': test_metrics[0]}
    plt.bar(list(data.keys()), list(data.values()))

    plt.subplot(2, 3, 2)
    data = {'train_MAE': train_metrics[1], 'val_MAE': val_metrics[1], 'test_MAE': test_metrics[1]}
    plt.bar(list(data.keys()), list(data.values()))

    plt.subplot(2, 3, 3)
    data = {'train_ARE': train_metrics[2], 'val_ARE': val_metrics[2], 'test_ARE': test_metrics[2]}
    plt.bar(list(data.keys()), list(data.values()))

    plt.subplot(2, 3, 4)
    data = {'train_ALE': train_metrics[3], 'val_ALE': val_metrics[3], 'test_ALE': test_metrics[3]}
    plt.bar(list(data.keys()), list(data.values()))

    plt.subplot(2, 3, 5)
    data = {'train_thresh_1.25': train_metrics[4], 'val_thresh_1.25': val_metrics[4], 'test_thresh_1.25': test_metrics[4]}
    plt.bar(list(data.keys()), list(data.values()))

def eval_metrics_iter(gt_depth_map, pred_depth_map):
    mse = RMSE(gt_depth_map, pred_depth_map) / gt_depth_map.shape[0] # BS Normalization
    mae = (MAE(gt_depth_map, pred_depth_map))  / gt_depth_map.shape[0] # BS Normalization
    are = (ARE(gt_depth_map, pred_depth_map))  / gt_depth_map.shape[0] # BS Normalization
    ale = (ALE(gt_depth_map, pred_depth_map))  / gt_depth_map.shape[0] # BS Normalization
    thresh125 = (thresh_acc(gt_depth_map, pred_depth_map))  / gt_depth_map.shape[0] # BS Normalization
    thresh156 = (thresh_acc(gt_depth_map, pred_depth_map, threshold=1.5625))  / gt_depth_map.shape[0] # BS Normalization
    thresh195 = (thresh_acc(gt_depth_map, pred_depth_map, threshold=1.953125))  / gt_depth_map.shape[0] # BS Normalization
    return (mse, mae, are, ale, thresh125, thresh156, thresh195)

def eval_metrics(y_true, y_pred): #, device):
    #evaluation of metrics over all batches
    mse, mae, are, ale, thresh125, thresh156, thresh195 = [], [], [], [], [], [], []
    for gt_depth_map, pred_depth_map in tqdm(zip(y_true, y_pred)):
        #images = images.cuda() #to(device)
        #gt_depth_map = gt_depth_maps.cuda() #to(device)
        #pred_depth_maps = net(images)
        mse.append(RMSE(gt_depth_map, pred_depth_map)) / gt_depth_map.shape[0] # BS Normalization
        mae.append(MAE(gt_depth_map, pred_depth_map)) / gt_depth_map.shape[0] # BS Normalization
        are.append(ARE(gt_depth_map, pred_depth_map)) / gt_depth_map.shape[0] # BS Normalization
        ale.append(ALE(gt_depth_map, pred_depth_map)) / gt_depth_map.shape[0] # BS Normalization
        thresh125.append(thresh_acc(gt_depth_map, pred_depth_map)) / gt_depth_map.shape[0] # BS Normalization
        thresh156.append(thresh_acc(gt_depth_map, pred_depth_map, threshold=1.5625)) / gt_depth_map.shape[0] # BS Normalization
        thresh195.append(thresh_acc(gt_depth_map, pred_depth_map, threshold=1.953125)) / gt_depth_map.shape[0] # BS Normalization
        
    mse = np.mean(mse)
    mae = np.mean(mae)
    are = np.mean(are)
    ale = np.mean(ale)
    thresh125 = np.mean(thresh125)
    thresh156 = np.mean(thresh156)
    thresh195 = np.mean(thresh195)
    print('\n', 'RMSE:', mse, 'MAE:', mae, 'ARE:', are, 'ALE:', ale, 'Thresh 1.25:', thresh125, 'Thresh 1.25^2:', thresh156, 'Thresh 1.25^3:', thresh195)
    return (mse, mae, are, ale, thresh125, thresh156, thresh195)

#plot_metrics(train_metrics, val_metrics, test_metrics)





