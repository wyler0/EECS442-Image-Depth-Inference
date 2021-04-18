"""
EECS 442 - Final Project

Code for training and validating and plotting a single epoch
"""

import torch
import numpy as np
import random
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
    for X, y in eval_loader:
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
    for i, (X, y) in enumerate(data_loader):
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

def _execute_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """

    # Setup and execute training
    y_true_train, y_pred_train, running_loss_train = _train_epoch(tr_loader, model, criterion)
    
    # Calculate train loss and accuracy for epoch
    train_loss = np.sum(running_loss_train) #TODO, use metrics!
    train_acc = 0 #TODO, Use metrics!
    
    # Setup and execute evaluation
    y_true_eval, y_pred_eval, running_loss_eval = _predict(val_loader, model, criterion)

    # Calculate val loss and accuracy for epoch
    val_loss = np.sum(running_loss_eval) #TODO, use metrics!
    val_acc = 0 #TODO, Use metrics!
    
    # Store data & plot
    stats.append([val_acc, val_loss, train_acc, train_loss])
    update_training_plot(axes, epoch, stats)
    return stats