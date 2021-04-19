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

def _predict(eval_loader, model, criterion):
    """
    Evaluates the `model` on the train and validation set.
    """
    # Put model in eval mode
    model.eval()
    
    # Setup and execute evaluation
    y_true, y_pred = [], []
    running_loss = []
    for X, y in tqdm(eval_loader):
        X.to(device=cuda)
        y.to(device=cuda)
        with torch.no_grad():
            output = model(X)
            y_pred.append(output)
            y_true.append(y)
            running_loss.append(criterion(output, y))

    # Return data
    return (y_true, y_pred, running_loss)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # Put model in train modee
    model.train()
    
    # Seetup and execute training
    y_true, y_pred = [], []
    running_loss = []
    for i, (X, y) in tqdm(enumerate(data_loader)):
        # convert inputs to correct type
        X = X.float().to(device=cuda)
        y.to(device=cuda)
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # store results
        y_pred.append(output)
        y_true.append(y)
        running_loss.append(criterion(output, y))

    # Return data
    return (y_true, y_pred, running_loss)    

def _execute_epoch(axes, tr_loader, val_loader, model, criterion, optimizer, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """

    # Setup and execute training
    y_true_train, y_pred_train, running_loss_train = _train_epoch(tr_loader, model, criterion, optimizer)
    
    # Evaluate metrics & loss
    tr_metrics = eval_metrics(y_true_train, y_pred_train) 
    train_loss = np.sum(running_loss_train) #TODO, use metrics!
    
    # Setup and execute evaluation
    y_true_eval, y_pred_eval, running_loss_eval = _predict(val_loader, model, criterion)

    # Evaluate metrics & loss
    va_metrics = eval_metrics(y_true_eval, y_pred_eval) 
    val_loss = np.sum(running_loss_eval) #TODO, use metrics!
    
    # Store data & plot
    stats.append([tr_metrics, train_loss, va_metrics, val_loss])
    update_training_plot(axes, epoch, stats)
    return stats