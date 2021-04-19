"""
EECS 442 - Final Project

Code for training and validating and plotting a single epoch
"""

import torch
import numpy as np
import random
from tqdm import tqdm
import gc

from metrics import eval_metrics, eval_metrics_iter
from utils import update_training_plot

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def _predict(eval_loader, model, criterion, use_cuda=False):
    """
    Evaluates the `model` on the train and validation set.
    """
    # Put model in eval mode
    model.eval()
    
    # Setup and execute evaluation
    running_loss = torch.zeros((len(eval_loader),))
    tr_metrics = torch.zeros((len(eval_loader), 7))
    for i, (X, y) in enumerate(tqdm(eval_loader)):
        X = X.float()
        if(use_cuda):
            X = X.to(device="cuda")
            y = y.to(device="cuda")
        with torch.no_grad():
            output = model(X)
            if(use_cuda):
                output = output.to(device="cuda")
            y = y.unsqueeze(3) # Add axis of dim 1 to end of gt for metrics
            output = output.permute(0,2,3,1) # Move channel axis for metrics
            running_loss[i] = criterion(output, y)
            tr_metrics[i] = torch.Tensor(list(eval_metrics_iter(y, output)))
        if(use_cuda):
            del X
            del y
            del output
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    # Return normalized statistics
    return (torch.sum(tr_metrics, axis=0)/len(eval_loader), torch.sum(running_loss)/len(eval_loader))

def _train_epoch(data_loader, model, criterion, optimizer, use_cuda=False):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # Put model in train modee
    model.train()
    
    # Seetup and execute training
    running_loss = torch.zeros((len(data_loader),))
    tr_metrics = torch.zeros((len(data_loader), 7))
    for i, (X, y) in enumerate(tqdm(data_loader)):
        #print("\n*******S1*******")
        #print(torch.cuda.memory_allocated())
        
        # convert inputs to correct type
        X = X.float()
        if(use_cuda):
            y = y.to(device="cuda")
            X = X.to(device="cuda")
        
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        if(use_cuda):
            output = output.to(device="cuda")
        loss = criterion(np.squeeze(output, axis=1), y)
        loss.backward()
        optimizer.step()

        # store results
        y = y.unsqueeze(3) # Add axis of dim 1 to end of gt for metrics
        output = output.permute(0,2,3,1) # Move channel axis for metrics
        tr_metrics[i] = torch.Tensor(list(eval_metrics_iter(y, output)))
        running_loss[i] = loss
        if(use_cuda):
            del X
            del y
            del output
            del loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    # Return normalized statistics
    return (torch.sum(tr_metrics, axis=0)/len(data_loader), torch.sum(running_loss)/len(data_loader))

def _execute_epoch(axis, tr_loader, val_loader, model, criterion, optimizer, epoch, stats, use_cuda=False):
    """
    Evaluates the `model` on the train and validation set.
    """

    # Setup and execute training
    tr_metrics, train_loss = _train_epoch(tr_loader, model, criterion, optimizer, use_cuda=use_cuda)
    
    # Setup and execute evaluation
    va_metrics, val_loss = _predict(val_loader, model, criterion, use_cuda=use_cuda)

    # Store data & plot
    stats.append([tr_metrics, train_loss, va_metrics, val_loss])
    #update_training_plot(axes, epoch, stats)
    return stats