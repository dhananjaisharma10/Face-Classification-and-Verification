import os
import time
import argparse
import numpy as _np

import torch
import torchvision as _tv
import torch.nn as _nn
import torch.nn.functional as _F

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import resnet50, init_weights


def arg_parser():
    parser = argparse.ArgumentParser(description='Take training and '
                                     'testing parameters')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate to train the model')
    parser.add_argument('--w_decay', type=float, default=5e-5,
                        help='weight decay for the optimizer')
    parser.add_argument('--train_data', require=True,
                        help='path to the training data')
    parser.add_argument('--val_data', required=True,
                        help='path to the validation data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for the dataloader')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train the network')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--device', default='cuda',
                        help='device to be used for training')
    parser.add_argument('--save_loc', default='./models',
                        help='directory to save models')
    return parser.parse_args()


# A run ID for a particular training session
def get_run_id():
    dt = datetime.now()
    run_id = dt.strftime('%m_%d_%H_%M')
    return run_id


# Train the network
def train_classify(model,
                   train_loader,
                   test_loader,
                   optimizer,
                   criterion,
                   num_epochs,
                   device,
                   save_loc):
    model.train()
    max_val_acc = 0

    # Validate the network
    def test_classify():
        model.eval()
        test_loss = list()
        accuracy = 0
        total = 0
        for (feats, labels) in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            _, pred_labels = torch.max(_F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)
            loss = criterion(outputs, labels.long())
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])
            del feats, labels
        return _np.mean(test_loss), accuracy/total

    for epoch in range(num_epochs):
        avg_loss = 0.0
        start_time = time.time()
        for batch_num, (feats, labels) in enumerate(train_loader):
            num_examples = len(train_loader)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)  # the labels
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            print('Iter = {}/{} | Training Loss = {:.3f}'.format(batch_num+1,
                  num_examples, (avg_loss/(batch_num+1))),
                  end="\r", flush=True)
            torch.cuda.empty_cache()
            del feats, labels, loss
        end_time = time.time()
        avg_loss /= num_examples
        print('\nTraining Loss: {} | Time: {:.0f} mins'.format(avg_loss,
              end_time-start_time)/60)
        val_loss, val_acc = test_classify()
        print('Val Loss: {:.4f} | Val Accuracy: {:.4f}'.format(val_loss,
              val_acc))
        # Save model if it scored the highest validation accuracy so far
        if max_val_acc < val_acc:
            torch.save(model.state_dict(),
                       os.path.join(save_loc, 'ckpt_{}.pth'.format(epoch)))
            max_val_acc = val_acc


def runner():
    args = arg_parser()
    run_id = get_run_id()
    save_loc = os.path.join(args.save_loc, run_id)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    device = None
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset = ImageFolder(root=args.train_data,
                                transform=_tv.transforms.ToTensor())
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    val_dataset = ImageFolder(root=args.val_data,
                              transform=_tv.transforms.ToTensor())
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    num_classes = len(train_dataset.classes)
    # Model
    network = resnet50(num_classes)
    network.apply(init_weights)
    # Loss function and Optimizer
    criterion = _nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.w_decay)
    network.to(device)
    # Training
    train_classify(network,
                   train_loader,
                   val_loader,
                   optimizer,
                   criterion,
                   args.epochs,
                   device,
                   save_loc)


if __name__ == '__main__':
    runner()
