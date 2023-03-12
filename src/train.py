import time
import os, copy, random
import itertools
import io
import datetime
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from config import args_parser
from models import *
from dataset import SkinCancer
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score

# import tensorflow as tf

import wandb
import warnings

warnings.filterwarnings('ignore')

# setting up LOGGER
import logging
from utils import setup_logging


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8, horizontalalignment='right')
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
                    
    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        val_correct += (predictions == labels).sum().item()

    #         y_true.extend(labels.cpu().numpy())
    #         y_pred.extend(predictions.cpu().numpy())

    #         y_t.append(labels.cpu().numpy())
    #         y_p.append(predictions.cpu().numpy())

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    #     cf_matrix = confusion_matrix(y_true, y_pred)

    #     f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_t, y_p)]
    #     p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_t, y_p)]
    #     r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_t, y_p)]
    #     # auc_l = [roc_auc_score(p.cpu().numpy(), t.cpu().numpy(), multi_class='ovr') for t,p in zip(y_true,y_pred)]

    #     f1_s = sum(f_l)/len(f_l)
    #     # print('f1:', f1_s)
    #     p_s = sum(p_l)/len(p_l)
    #     r_s = sum(r_l)/len(r_l)

    #     df = {'F1_Score' : f1_s,
    #           'Precision' : p_s,
    #           'Recall' : r_s}

    #     metrics_table.add_data(data=df)

    #     wandb.log({"TrainingConfusionMatrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds = y_pred, class_names = class_names)})

    return valid_loss, val_correct


def test_inference(model, device, dataloader, loss_fn, class_names):
    test_loss, test_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        test_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        test_correct += (predictions == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

        y_t.append(labels.cpu().numpy())
        y_p.append(predictions.cpu().numpy())

    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_figure = plot_confusion_matrix(cf_matrix, class_names)

    #     f_l = [f1_score(p, t, average='macro') for t,p in zip(y_t, y_p)]
    #     p_l = [precision_score(p, t, average='macro') for t,p in zip(y_t, y_p)]
    #     r_l = [recall_score(p, t, average='macro') for t,p in zip(y_t, y_p)]
    #     # auc_l = [roc_auc_score(p.cpu().numpy(), t.cpu().numpy(), multi_class='ovr') for t,p in zip(y_true,y_pred)]

    #     f1_s = sum(f_l)/len(f_l)
    #     # print('f1:', f1_s)
    #     p_s = sum(p_l)/len(p_l)
    #     r_s = sum(r_l)/len(r_l)

    # m_dict = pd.DataFrame({'F1_Score' : f1_s,
    #       'Precision' : p_s, 
    #       'Recall' : r_s})

    # metrics_table.add_data(f1_s,p_s,r_s)

    #     wandb.log({"Testing-Confusion-Matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds = y_pred, class_names = class_names)})

    #     wandb.log({"Metrics-Table": wandb.Table(columns=['F1_Score','Precision','Recall'], data=[[f1_s, p_s, r_s]])})

    return test_loss, test_correct, cf_figure, cf_matrix


if __name__ == '__main__':

    args = args_parser()

    # Set device parameter
    if args.gpu:
        if os.name == 'posix' and torch.backends.mps.is_available():  # device is mac m1 chip
            # LOGGER.info(f"Device Type : Using MPS")
            device = 'mps'
        elif os.name == 'nt' and torch.cuda.is_available():  # device is windows with cuda
            # LOGGER.info(f'Device Type : Using CUDA')
            device = args.device
        else:
            # LOGGER.info(f'Device Type : Using CPU')
            device = 'cpu'

    # Initialize metrics table to log metrics to wandb

    # metrics_table = wandb.Table(columns=['F1_Score','Precision','Recall'])

    # ======================= DATA ======================= #

    data_dir = '../data/Combined_data/'
    dataset = SkinCancer(data_dir, '../data/train.csv', transform=None)
    dataset_size = len(dataset)
    test_dataset = SkinCancer(data_dir, '../data/test.csv', transform=None)
    classes = np.unique(dataset.classes)

    # ======================= Model | Loss Function | Optimizer ======================= #

    if args.model == 'efficientnet':
        model = efficientnet()

        # if args.finetune:
        #     model.classifier = nn.Sequential(
        #     nn.BatchNorm1d(num_features=1280),    
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, num_features),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_features=1280),
        #     nn.Dropout(0.4),
        #     nn.Linear(1280, 7),
        #     )


    elif args.model == 'resnet':
        model = resnet()

    elif args.model == 'vit':
        model = vit()

    elif args.model == 'convnext':
        model = convnext()

    elif args.model == 'alexnet':
        model = alexnet()

    elif args.model == 'cnn':
        model = cnn()

    # copy weights
    MODEL_WEIGHTS = copy.deepcopy(model.state_dict())

    # ======================= Set Optimizer and loss Function ======================= #
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9)
    elif args.optimizer == 'adamx':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    if args.imbalanced:
        # loss function with class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(dataset.classes),
                                                          y=np.array(dataset.classes_all))
        class_weights = torch.FloatTensor(class_weights).cuda()
        # class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(dataset.classes),y=self.df['dx'].to_numpy()),device='cuda')
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')


    else:
        criterion = nn.CrossEntropyLoss()

    batch_size = args.batch
    class_names = dataset.classes

    # ======================= Logger ======================= #      

    if args.logger == 'tb':

        logger = SummaryWriter(log_dir=f'../tb_logs/{model._get_name()}/{args.modality}_{args.epochs}Epochs')

    elif args.logger == 'wb':
        wandb.login(key="7a2f300a61c6b3c4852452a09526c40098020be2")
        logger = wandb.init(
            # Set the project where this run will be logged
            project="SkinCancer_Augmented_CV_UpdateWeights", entity="fau-computer-vision",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": "Skin Cancer",
                "epochs": args.epochs
            })

    else:
        logger = None

    # ======================= Start ======================= #
    start_t = time.time()
    best_acc = 0.0
    step = 0
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    args.finetune = 'finetune' if args.finetune else 'transfer'

    # ======================= Local Logger ======================= #

    exp_dir = f'../tb_logs/logs/{model._get_name()}_{args.epochs}/'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/log.log"
    LOGGER = logging.getLogger(__name__)
    setup_logging(log_path=log_file, log_level='INFO', logger=LOGGER)

    # ======================= Local Logger ======================= #
    LOGGER.info(f'Device: {device}')
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        LOGGER.info('Fold: {}, Model: {}'.format(fold, model._get_name()))

        # model.load_state_dict(MODEL_WEIGHTS) # uncomment to start fresh for each fold

        model.to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler)  # train, will change for each fold
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)  # validation
        test_loader = DataLoader(test_dataset, batch_size=batch_size)  # hold out set, test once at the end of each fold

        # ======================= Train per fold ======================= #
        for epoch in range(args.epochs):
            # print(f'Epoch :: {epoch}')
            step += 1
            train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_correct = valid_epoch(model, device, val_loader, criterion)
            test_loss_epoch, test_acc_epoch, cf_figure, _ = test_inference(model, device, test_loader, criterion,
                                                                           class_names)
            logger.add_figure("Confusion Matrix Epoch", cf_figure, step)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            val_loss = val_loss / len(val_loader.sampler)
            val_acc = val_correct / len(val_loader.sampler) * 100

            # print(f"Epoch: {epoch}/{args.epochs},\n AVG Training Loss:{train_loss} \t Validation Loss{val_loss}\nAVG
            # Training Acc: {train_acc} % \t Validation Acc {val_acc}")
            LOGGER.info(f'Epoch: {epoch}/{args.epochs}')
            LOGGER.info(f'Average Training Loss: {train_loss}')
            LOGGER.info(f'Average Validation Loss: {val_loss}')
            LOGGER.info(f'Average Training Acc: {train_acc}')
            LOGGER.info(f'Average Validation acc: {val_acc}')

            test_loss_epoch = test_loss_epoch / len(test_loader.sampler)
            test_acc_epoch = test_acc_epoch / len(test_loader.sampler) * 100

<<<<<<< Updated upstream
            # print("Epoch:{}/{}\nAVG Training Loss:{:.3f} \t Testing Loss:{:.3f}\nAVG Training Acc: {:.2f} % \t Testing Acc {:.2f} % ".format(epoch, args.epochs, train_loss,  val_loss, train_acc,  val_acc))
=======

            # print("Epoch:{}/{}\nAVG Training Loss:{:.3f} \t Testing Loss:{:.3f}\nAVG Training Acc: {:.2f} % \t Testing
            # Acc {:.2f} % ".format(epoch, args.epochs, train_loss,  val_loss, train_acc,  val_acc))
            
>>>>>>> Stashed changes
            # ======================= Save per Epoch ======================================= #

            logger.add_scalars('Loss', {'train': train_loss,
                                        'val': val_loss,
                                        'test': test_loss_epoch}, step)

            logger.add_scalars('Acc', {'train': train_acc,
                                       'val': val_acc,
                                       'test': test_acc_epoch}, step)

            # ======================= Save model if new high accuracy ======================= #
            if test_acc_epoch > best_acc:
                LOGGER.info(f'New High Acc: <<<<< {test_acc_epoch} >>>>>')

                best_acc = test_acc_epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           f'../models/{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pth')

                # Save Scripted Model 
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model,
                               f'../models/scripted_{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pt')

        # ======================= Test Model on HOS ======================= #

<<<<<<< Updated upstream
        test_loss, test_correct, cf_figure_fold, cf_matrix = test_inference(model, device, test_loader, criterion,
                                                                            class_names)
=======
        # ======================= Test Model on HOS ======================= #

        test_loss, test_correct, cf_figure_fold, cf_matrix = test_inference(model,device,test_loader,criterion,class_names)
>>>>>>> Stashed changes

        logger.add_figure("Confusion Matrix Fold", cf_figure_fold, fold)

        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

<<<<<<< Updated upstream
        np.save(f'../output_files/cf_matrix/{model._get_name()}_{args.modality}_{args.finetune}_Fold{fold}.npy',
                cf_matrix)
=======
        np.save(f'../output_files/cf_matrix/{model._get_name()}_{args.modality}_{args.finetune}_Fold{fold}.npy', cf_matrix)
>>>>>>> Stashed changes

        # print("Fold:{}/{}\nTesting Loss:{:.3f} \t Testing Acc:{:.3f}% ".format(fold,test_loss, test_acc))
        # print(f"Fold:{fold}\nTesting Loss:{test_loss} \t Testing Acc:{test_acc}%")
        # wandb.log({"Fold Test": {"test_loss" : test_loss,
        #                          "test_acc" : test_acc}})

        logger.add_scalar('Fold/Acc', test_acc, fold)
        logger.add_scalar('Fold/Loss', test_loss, fold)

<<<<<<< Updated upstream
        #

=======
>>>>>>> Stashed changes
        # ======================= Save model if new high accuracy ======================= #
        if test_acc > best_acc:
            # print('#'*25)
            LOGGER.info(f'New High Acc: <<<<< {test_acc} >>>>>')
            # print('#'*25)
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       f'../models/{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pth')

            # Save Scripted Model 
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model,
                           f'../models/scripted_{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pt')

    end_train = time.time()
    time_elapsed = start_t - end_train

    LOGGER.info(f'{model._get_name()} Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
