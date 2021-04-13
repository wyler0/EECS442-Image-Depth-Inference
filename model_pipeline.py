'''
EECS 498 - Machine Learning for Affective Computing
Winter 2021 - Final Project
Train, validate, and predict model
    Runs everything
    Usage: python modelPipe.py
'''
import torch
import itertools
import numpy as np

from dataset import get_train_val_test_loaders # TODO, add this function / properly load form dataset loader
from models.basemodel import BaseModel # TODO, update this to the actual model
from train_model import _train_epoch, _evaluate_epoch, _predict
from utils import get_hyper_parameters, make_training_plot, save_training_plot, config

def main():
    # Data loaders
    tr_loader, va_loader, te_loader = get_train_val_test_loaders()
    model = BaseModel()


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
        
        
        stats = []

        fig, axes = make_training_plot('basemodel')

        # Loop over the entire dataset multiple times
        for epoch in range(0, config('basemodel.num_epochs')):
            # Train model
            _train_epoch(tr_loader, model, criterion, optimizer)

            # Evaluate model
            _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch, stats)

        print('Finished Training')
        """
        # TODO, Update these to use the new metrics
        print("Validation Accuracy \t Validation Loss \t Training Accuracy \t Training Loss\n")

        for stat in stats:
            val_acc, val_loss, train_acc, train_loss = stat
            print(val_acc, val_loss, train_acc, train_loss, sep='\t')
        """

        print('Begin model evaluation...')

        # Evaluate model
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
        print('Finished Model Evaluation')

        save_training_plot(fig, 'basemodel')

    print("Best learning rate: {}, best weight_decay: {}".format(best_lr, best_wd))
    print("Weighted F-1: {:.4f}".format(best_wf1))

    utils.hold_training_plot()

if __name__ == '__main__':
    main()