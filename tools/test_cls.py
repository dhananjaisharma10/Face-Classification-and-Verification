import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import resnet50
from dataset import ImageDataset, parse_data


def arg_parser():
    parser = argparse.ArgumentParser(description='Testing parameters'
                                     ' for Classification')
    parser.add_argument('--test_data', required=True,
                        help='path to the testing data')
    parser.add_argument('--run_id', required=True,
                        help='test models under this run ID')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='total number of classes')
    parser.add_argument('--models', required=True, nargs='+',
                        help='models to be used for testing')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--device', default='cuda',
                        help='device to be used for training')
    parser.add_argument('--save_loc', default='./models',
                        help='directory to save models')
    return parser.parse_args()


def test_cls(models,
             test_loader,
             num_classes,
             device):
    """Takes an ensemble of models as input
    and outputs the predicted labels.
    """
    labels = list()
    with torch.no_grad():
        for feats in test_loader:
            feats = feats.to(device)
            temp = torch.zeros(len(feats), num_classes).to(device)
            for model in models:
                model.eval()
                model.to(device)
                temp += F.softmax(model(feats), dim=1)
            temp /= len(models)
            _, pred_labels = torch.max(temp, dim=1)
            pred_labels = pred_labels.view(-1)
            labels.append(pred_labels.cpu())
            del feats
    return labels


def get_csv(labels, save_loc):
    # Generating the CSV
    a = pd.DataFrame(np.concatenate(labels), columns=["label"])
    a["id"] = a.index
    a = pd.concat([a['id'], a['label']], axis=1)  # Rearranging the columns
    b = pd.read_csv('true_class_labels.csv', header=None)
    b.drop(columns=[0], axis=1, inplace=True)
    for x in range(len(a['label'])):
        a['label'][x] = b.iloc[a['label'][x], 0]
    a.to_csv(os.path.join(save_loc, 'results_cls.csv'), index=False)


def evaluate():
    args = arg_parser()
    save_loc = os.path.join(args.save_loc, args.run_id)
    if not os.path.exists(save_loc):
        raise AssertionError('No directory named'
                             ' as {} found'.format(save_loc))
    network = resnet50(args.num_classes)
    models = list()
    # Ensemble of models (a single model can also be passed for testing)
    for model in args.models:
        network.load_state_dict(torch.load(os.path.join(save_loc, model)))
        models.append(network)

    device = None
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    img_list = parse_data(args.test_data)
    # NOTE: You may sort the data according to your usecase
    img_list.sort(key=lambda x: int(x[39:].strip('.jpg')))
    # Dataset and Dataloader
    test_dataset = ImageDataset(img_list)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    assigned_labels = test_cls(models,
                               test_loader,
                               args.num_classes,
                               device)
    # NOTE: You may amend this function for your usecase
    get_csv(assigned_labels, save_loc)


if __name__ == '__main__':
    evaluate()
