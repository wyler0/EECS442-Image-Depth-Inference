"""
EECS 442 - Final Proj
Functions for model loading and saving and config interaction
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def get_hyper_parameters(modType):
    # Get lr and wd
    lr = config(modType+'.learning_rate')
    weight_decay = config(modType+'.weight_decay')

    return lr, weight_decay

def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

def make_training_plot(modType):
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    # plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle(modType + ' Model Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    return fig, axes

def update_training_plot(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    train_loss = [s[3] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0].legend(['Validation', 'Train'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[1].legend(['Validation', 'Train'])
    # plt.pause(0.00001)

def save_training_plot(fig, modType):
    """
    Saves the training plot to a file
    """
    fig.savefig('plots/' + modType + '_model_training_plot.png', dpi=200)

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    # plt.ioff()
    plt.show()
