import os
import argparse
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataset import MyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def arg_parser():
    parser = argparse.ArgumentParser(description='Testing parameters'
                                     ' for Verification')
    parser.add_argument('--test_file', required=True,
                        help='path to the txt file containing pairs of'
                        ' images and their auc score (only for validation)')
    parser.add_argument('--vrf_file', default='./vrf.csv',
                        help='path to the csv file containing pairs of'
                        ' images to be compared')
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


def test_vrf(models,
             test_loader,
             num_classes,
             device,
             val_labels=None):
    """Takes an ensemble of models as input
    and outputs the cosine similarity scores.
    """
    similarity = list()
    with torch.no_grad():
        for feats in test_loader:
            img1, img2 = feats
            img1, img2 = img1.to(device), img2.to(device)
            temp = torch.zeros(len(feats[0])).to(device)
            for model in models:
                model.eval()
                model.to(device)
                output_1, output_2 = model(img1)[0], model(img2)[0]
                temp += F.cosine_similarity(output_1, output_2)
                torch.cuda.empty_cache()
            temp /= len(models)
            similarity = similarity + temp.tolist()
            torch.cuda.empty_cache()
            del temp, img1, img2, feats

    if val_labels is not None:
        return roc_auc_score(val_labels, similarity)
    else:
        return similarity


def get_csv(vrf_file, similarity, save_loc):
    f = pd.read_csv(vrf_file)
    f['score'] = similarity
    f.to_csv(os.path.join(save_loc, 'results_vrf.csv'), index=False)


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
    # Dataset and Dataloader
    test_dataset = MyDataset(args.test_file,
                             args.test_data,
                             args.mode,
                             transforms.ToTensor())
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    # Testing
    if args.mode == 'Testing':
        similarity = test_vrf(models,
                              test_loader,
                              args.num_classes,
                              device,
                              None)
        get_csv(args.vrf_file, similarity, save_loc)
    # Validation
    else:
        val_labels = test_dataset.getLabels()
        auc_score = test_vrf(models,
                             test_loader,
                             args.num_classes,
                             device,
                             val_labels)
        print('AUC Score for Validation: {:.3f}'.format(auc_score))


if __name__ == '__main__':
    evaluate()
