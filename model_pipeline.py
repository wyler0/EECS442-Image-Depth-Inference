'''
EECS 442 - Computer Vision
Winter 2021 - Final Project
Train, validate, and predict model
    Runs everything
    Usage: python modelPipe.py
'''
import torch
import itertools
import numpy as np

from diode import DIODE, plot_depth_map
from model.model import ImageDepthPredModel
from train_model import _train_epoch, _evaluate_epoch, _predict
from utils import get_hyper_parameters, make_training_plot, save_training_plot, config
from torch.utils.data import DataLoader 
from torchsummary import summary

def main():
    # Dataset Metadata
    meta_fname = './diode_meta.json'
    data_root = './dataset'
    # Data loaders
    # indoors training data
    tr_indoors_data = DIODE(meta_fname, data_root, splits='train', scene_types='outdoor')
    va_indoors_data = DIODE(meta_fname, data_root, splits='val', scene_types='outdoor')
    te_indoors_data = DIODE(meta_fname, data_root, splits='test', scene_types='outdoor')
    # TODO: outdoor training data
    tr_loader = DataLoader(tr_indoors_data, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_indoors_data, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_indoors_data, batch_size=64, shuffle=True)
    model = ImageDepthPredModel()

    # Grab hyperparameters for model specified
    learning_rate, weight_decay = get_hyper_parameters('basemodel')

    # Get the best model return
    best_wd = 0
    best_lr = 0
    best_wf1 = 0

    # Grid search on weight decay and learning rate options
    for lr, wd in itertools.product(learning_rate, weight_decay):
        print('Training and evaluating Basemodel with: \tLR = ' +str(lr) + '\tWD = '+str(wd))

        # Define loss function, and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
<<<<<<< HEAD
=======
        # Setup plots and metrics results storage
>>>>>>> 9aaab9a1fe3b3958b8388df8d7a5ed002b09c0b0
        stats = []
        fig, axes = make_training_plot('basemodel')

        # Executee configured number of epochs training + validating
        for epoch in range(0, config('basemodel.num_epochs')):
            # Train model
            (tr_loader, model, criterion, optimizer)

            # Evaluate model
            stats = _execute_epoch(axes, tr_loader, va_loader, model, criterion, epoch, stats)

        print('Finished Training')
        
        """
        # TODO, Update these to use the new metrics
        print("Validation Accuracy \t Validation Loss \t Training Accuracy \t Training Loss\n")

        for stat in stats:
            val_acc, val_loss, train_acc, train_loss = stat
            print(val_acc, val_loss, train_acc, train_loss, sep='\t')
        """

        print('Begin model evaluation...')

        # Test model
        labels, preds = _predict(te_loader, model)

        """
        # TODO, Update these to use the new metrics
        acc_score = metrics.accuracy_score(correct_labels, model_pred)
        prec_score = metrics.precision_score(correct_labels, model_pred)
        rec_score = metrics.recall_score(correct_labels, model_pred)
        f1_score = metrics.f1_score(correct_labels, model_pred)
        conf_matrix = metrics.confusion_matrix(correct_labels, model_pred)
        print("accuracy score:", acc_score)
        print("f1 score:", f1_score)
        print("precision score:", prec_score)
        print("recall score:", rec_score)
        print("confusion matrix:", conf_matrix)

        if f1_score > best_wf1:
            best_wf1 = f1_score
            best_lr = lr
            best_wd = wd
        """
        print('Finished Model Testing')

        save_training_plot(fig, 'basemodel')

    print("Best learning rate: {}, best weight_decay: {}".format(best_lr, best_wd))
    print("Weighted F-1: {:.4f}".format(best_wf1))

    hold_training_plot()

if __name__ == '__main__':
    main()