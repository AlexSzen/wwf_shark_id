#!/usr/bin/env python3

'''
Training script for low to high frequency seismic data.
'''

import sys
from torch.optim import lr_scheduler
# helper I/O functions
from io_utils import clear_dir, createFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
import json
import random

# Helper training function
def train_model(model,
                dataloaders,
                device,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                targetDir):

    train_loss_list = []
    val_loss_list = []

    i = 0
    start = time.time()
    for iepoch in range(num_epochs):

        print('Starting epoch %i of %i ..' % (iepoch + 1, num_epochs))

        for phase in ['train', 'val']:

            print("Running %s phase.." % (phase))

            # set the model in training or evaluation state
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0
            for ibatch, (inputs, labels) in enumerate(dataloaders[phase]):
                #pdb.set_trace()
                # train step
                if phase == "train":
                    i = len(dataloaders[phase])*iepoch + ibatch
                # put inputs and outputs on device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the param gradients
                optimizer.zero_grad()

                # forward, track only for train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # divide by batch size (normalise)
                running_loss += loss.item() / inputs.shape[0]

                # print batch statistics
                if (i + 1) % 100 == 0 and phase == "train":
                    header = "[epoch: %i/%i batch: %i/%i i: %i]" % (iepoch + 1, num_epochs, ibatch + 1, len(dataloaders[phase]), i + 1)
                    print(header+" %s running loss: %.5f" % (phase, (running_loss / (ibatch + 1))))
                    print(header+" seconds / batch: %.4f" % ((time.time()-start)/100))
                    start = time.time()

            epoch_loss = running_loss / len(dataloaders[phase])

            print('Epoch %i complete. %s average loss = %f' % (iepoch + 1, phase, epoch_loss))

            # save epoch loss
            if phase == 'train':
                train_loss_list.append(epoch_loss)
                np.save(targetDir+"train_loss.npy", np.array(train_loss_list))
            elif phase == 'val':
                val_loss_list.append(epoch_loss)
                np.save(targetDir+"val_loss.npy", np.array(val_loss_list))

            # Save model every 2 epochs
            if (iepoch+1) % 20 == 0:

                # save model
                print("Saving model..")
                # put model on cpu before saving
                model.to(torch.device('cpu'))
                torch.save({
                'i' : i + 1,
                'model': model,
                }, targetDir+"models/"+"model_%.8i.torch" % (i + 1))
                model.to(device)

                # save 1 batch of output predictions on validation
                iterator = iter(dataloaders[phase])
                inputs, labels = next(iterator)

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                np.save(targetDir+"predictions/"+"inputs_%s_%.8i.npy" % (phase, i + 1),
                        inputs.detach().cpu().numpy())
                np.save(targetDir+"predictions/"+"outputs_%s_%.8i.npy" % (phase, i + 1),
                        outputs.detach().cpu().numpy())
                np.save(targetDir+"predictions/"+"labels_%s_%.8i.npy" % (phase, i + 1),
                        labels.detach().cpu().numpy())
                del iterator

    return model, np.array(train_loss_list), np.array(val_loss_list)


def parse_args():
    '''
    Source is the directory with json config files
    Target is the directory where results will be dumped
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src', required=False,
                        default="/scratch/lina3003/wwf_shark_id/config_files/config_2/")
    parser.add_argument('--target', dest='target', required=False,
                        default="/scratch/lina3003/results/sharks/resnet50ft_7cls/")
    parser.add_argument('--device', dest='device', required=False, default="0")
    parser.add_argument('--data_root', dest='data_root', required = False, default="/scratch/lina3003/data/sharks/data/")
    args = parser.parse_args()
    return args

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == "__main__":



    # get command line arguments
    args = parse_args()

    # get configuration file names, shuffle
    if not os.path.exists(args.target):
        os.mkdir(args.target)
    cfgs = [fn for fn in os.listdir(args.src) if fn.endswith(".json")]
    cfgs.sort()
    random.shuffle(cfgs)

    for cfgi, cfgPath in enumerate(cfgs):

        # get configuration file
        cfgName = cfgPath.replace(".json", "")
        cfg = json.load(open("%s/%s" % (args.src, cfgPath)))
        print()
        print("cfg:")
        for k in cfg:
            print("%s: %s" % (k, cfg[k]))
        print()

        # sort out output directories
        targetDir = "%s/%s/" % (args.target, cfgName)
        if not os.path.exists(targetDir):
            os.mkdir(targetDir)
        else:  # skip if output directory exists
            print("Skipping %s, directory exists" % (targetDir))
            continue
        createFolder(targetDir+"models/")
        createFolder(targetDir+"predictions/")

        # define CUDA device name
        device = torch.device("cuda:%i" % (int(args.device)) if torch.cuda.is_available() else "cpu")
        print("Device: ", device)
        print()

        # set random seed (for repeatability)
        torch.manual_seed(123)

        # define loss
        criterion = nn.CrossEntropyLoss() 
	
        model_name = "resnet"
        num_classes = 7
        feature_extract = True
	# Initialize the model for this run
        model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

	# Data augmentation and normalization for training
	# Just normalization for validation
        data_transforms = {
                'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	        ]),
	        'val': transforms.Compose([
	        transforms.Resize(input_size),
	        transforms.CenterCrop(input_size),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	        ]),
        }

        print("Initializing Datasets and Dataloaders...")



	# Create training and validation datasets
        datasets = {x: datasets.ImageFolder(os.path.join(args.data_root, x), data_transforms[x]) for x in ['train', 'val']}
	# Create training and validation dataloaders
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=cfg["batch_size"], shuffle=True, num_workers=0) for x in ['train', 'val']}



        # put model on device
        model = model.to(device)
        print(model)

        # define optimiser
        n_epochs = 100
        # optimizer = torch.optim.SGD(model.parameters(), lr=initLR, momentum=0.9,
        # nesterov = True, weight_decay = weightDecay)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg['lr_start'],
                                     weight_decay=cfg['weight_decay'])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                               step_size=cfg['lr_step_size'],
                                               gamma=1.)

        print(optimizer)

        # train
        start = time.time()
        model, train_loss, val_loss = train_model(model,
                                                  dataloaders,
                                                  device,
                                                  criterion,
                                                  optimizer,
                                                  exp_lr_scheduler,
                                                  n_epochs,
                                                  targetDir)
        delta = time.time() - start
        print("Training complete")

        # save log file
        logFile = "%s/%s_log.txt" % (targetDir, cfgName)
        F = open(logFile, 'w')
        F.write('Info and hyperparameters for ' + cfgName + '\n')
        F.write('Loss : ' + str(criterion) + '\n')
        F.write('Optimizer : ' + str(optimizer) + '\n')
        F.write('Scheduler : ' + str(exp_lr_scheduler) + ', step_size : ' + str(exp_lr_scheduler.step_size)+', gamma : ' + str(exp_lr_scheduler.gamma)+'\n')
        F.write('Number of epochs : ' + str(n_epochs) + '\n')
        F.write('Batch size : ' + str(cfg["batch_size"]) + '\n')
        F.write('Training time : ' + str(delta)+'s')
        F.write('Net architecture : \n' + str(model))
        F.close()
