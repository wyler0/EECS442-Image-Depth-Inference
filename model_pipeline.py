'''
EECS 442 - Computer Vision
Winter 2021 - Final Project
Train, validate, and predict model
    Runs everything
    Usage: python modelPipe.py
'''
import math
import itertools

import numpy as np
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from matplotlib import pyplot as plt

from diode import DIODE
from models.loss import ThreePartLoss
from models.model import ImageDepthPredModel
from train_model import _execute_epoch, _predict
from metrics import eval_metrics
from utils import get_hyper_parameters, config, make_training_plot, save_training_plot, hold_training_plot

def main(use_cuda=False, batch_size=16):
    if(use_cuda):
        print("Using GPU.")
    print("Batch size: " + str(batch_size))
    print("Epochs: " + str(config('basemodel.num_epochs')))
    # Setup & Split Datase
    dataset = DIODE('diode/diode_meta.json', 'diode', ['val'], ['indoors','outdoor'])
    indices = np.arange(0,len(dataset))
    np.random.shuffle(indices) # shuffle the indicies

    tr_split_ind = math.floor(0.7*len(dataset)) # 70% Train
    va_split_ind = math.floor(0.85*len(dataset)) # 15% Validation and 15% Test
    tr_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(indices[:tr_split_ind]), pin_memory=True)
    va_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(indices[tr_split_ind:va_split_ind]), pin_memory=True)
    te_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(indices[va_split_ind:]))

    # Load model
    model_name = "basemodel"
    model = ImageDepthPredModel()
    if(use_cuda):
        print("Model sent to GPU")
        model.cuda()
    save_dir = 'models/'

    # Grab hyperparameters for model specified
    learning_rate, weight_decay = get_hyper_parameters(model_name)

    # Get the best model return
    best_wd = 0
    best_lr = 0
    best_mse = 0

    # Grid search on weight decay and learning rate options
    for lr, wd in itertools.product(learning_rate, weight_decay):
        print('\nTraining and Evaluating Basemodel with: \tLR = ' +str(lr) + '\tWD = '+str(wd))

        # Define loss function, and optimizer
        criterion = ThreePartLoss(use_cuda=use_cuda)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        # Setup plots and metrics results storage
        stats = []
        fig, axes = make_training_plot('basemodel')

        # Executee configured number of epochs training + validating
        for epoch in range(0, config('basemodel.num_epochs')):
            print('\nEpoch #' + str(epoch))
            # Train model + Evaluate Model
            stats = _execute_epoch(axes, tr_loader, va_loader, model, criterion, optimizer, epoch, stats, use_cuda=use_cuda)
            
            if epoch%4 == 0: # Save every five epcoh's
                print("Saving model state and plots.")
                save_training_plot(fig, 'basemodel')
                check = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
                torch.save(check, save_dir+model_name+ "_save_" + str(epoch) + ".pt")
           
            train_acc = stats[len(stats)-1][0][0] #MSE
            train_loss = stats[len(stats)-1][1] #Loss
            val_acc = stats[len(stats)-1][2][0] #MSE
            val_loss = stats[len(stats)-1][3] #Loss
            print("\nTraining MSE\tTraining Loss\tValidation MSE\tValidation Loss\n")
            print(train_acc, train_loss, val_acc, val_loss, sep='\t')
            plt.close()
            plt.show()

            

        print('\nFinished Training. Saving plot....')
        save_training_plot(fig, 'basemodel')
        print('\nBegin Model Test Set Evaluation...')

        # Test model
        te_metrics, te_loss = _predict(te_loader, model, criterion, use_cuda=use_cuda)
        
        if te_metrics[0] > best_mse:
            best_mse = te_metrics[0]
            best_lr = lr
            best_wd = wd

        print("\nTesting MSE\tTesting Loss\n")
        print(te_metrics[0], te_loss, sep='\t')
        print('Finished Model Testing, Saving Model')
        check = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(check, save_dir+model_name+ "_save_final.pt")

    print("Best learning rate: {}, best weight_decay: {}".format(best_lr, best_wd))
    print("Best MSE: {:.4f}".format(best_mse))

    hold_training_plot()

if __name__ == '__main__':
    main()