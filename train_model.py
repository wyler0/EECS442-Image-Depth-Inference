"""
EECS 442 - Final Project

Code for training and validating and plotting a single epoch
"""

import torch
import numpy as np
import random
from tqdm import tqdm

from metrics import eval_metrics
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
    y_true, y_pred = [], []
    running_loss = []
    for X, y in tqdm(eval_loader):
        X = X.float()
        if(use_cuda):
            X = X.to(device="cuda")
            y = y.to(device="cuda")
        with torch.no_grad():
            output = model(X)
            if(use_cuda):
                output = output.to(device="cuda")
            running_loss.append(criterion(np.squeeze(output), y))
            y_pred.append(output.permute(0,2,3,1)) # Move channel axis for metrics
            y_true.append(y.unsqueeze(2)) # Add axis of dim 1 to end of gt for metrics

    # Return data
    return (y_true, y_pred, running_loss)

def _train_epoch(data_loader, model, criterion, optimizer, use_cuda=False):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # Put model in train modee
    model.train()
    
    # Seetup and execute training
    y_true, y_pred = [], []
    running_loss = []
    print(len(data_loader))
    for i, (X, y) in tqdm(enumerate(data_loader)):
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
        y = y.unsqueeze(3)
        output = output.permute(0,2,3,1)
        running_loss.append(criterion(output, y))
        y_pred.append(output) # Move channel axis for metrics
        y_true.append(y) # Add axis of dim 1 to end of gt for metrics
        del X
        del y
        torch.cuda.empty_cache()

    # Return data
    return (y_true, y_pred, running_loss)    

def _execute_epoch(axis, tr_loader, val_loader, model, criterion, optimizer, epoch, stats, use_cuda=False):
    """
    Evaluates the `model` on the train and validation set.
    """

    # Setup and execute training
    y_true_train, y_pred_train, running_loss_train = _train_epoch(tr_loader, model, criterion, optimizer, use_cuda=use_cuda)
    
    # Evaluate metrics & loss
    tr_metrics = eval_metrics(y_true_train, y_pred_train) 
    train_loss = np.mean(running_loss_train) #TODO, use metrics!
    
    # Setup and execute evaluation
    y_true_eval, y_pred_eval, running_loss_eval = _predict(val_loader, model, criterion, use_cuda=use_cuda)

    # Evaluate metrics & loss
    va_metrics = eval_metrics(y_true_eval, y_pred_eval) 
    val_loss = np.mean(running_loss_eval) #TODO, use metrics!
    
    # Store data & plot
    stats.append([tr_metrics, train_loss, va_metrics, val_loss])
    #update_training_plot(axes, epoch, stats)
    return stats