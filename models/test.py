import sys
sys.path.append('../')
import os
import argparse
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from sklearn.metrics import precision_score, recall_score, f1_score
from models import network
from utils.dataset import CatDogDataset
from models import configs


def cal_metrics(y_true, y_pred):

    data_len = len(y_pred)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(data_len):
        if int(y_true[i]) == 1 and int(y_pred[i]) == 1:
            TP += 1
        elif int(y_true[i]) == 1 and int(y_pred[i]) == 0:
            FN += 1
        elif int(y_true[i]) == 0 and int(y_pred[i]) == 1:
            FP += 1
        elif int(y_true[i]) == 0 and int(y_pred[i]) == 0:
            TN += 1
        else:
            print("Error: Some error in category")

    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    TPR = TP/(TP+FN)
    P = TP/(TP+FP)
    F1 = 2*P*TPR/(P+TPR)

    return FP, FN, TP, TN, FPR, FNR, TPR, P, F1


def test(args):

    config = getattr(configs, args.model + 'Config')()
    model = getattr(network, args.model)(config).eval()

    test_set = CatDogDataset(root_path=config.test_path, config=config, mode='test')
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    y_true = []
    y_pred = []
    test_confusion_matrix = meter.ConfusionMeter(config.num_classes)
    test_confusion_matrix.reset()

    model.eval()
    for _iter, (test_data, test_label) in enumerate(test_dataloader):

        if args.use_gpu:
            test_data = test_data.cuda()

        test_logits, test_output = model(test_data)
        y_true.extend(test_label.numpy().tolist())
        y_pred.extend(test_logits.max(dim=1)[1].detach().tolist())
        test_confusion_matrix.add(test_logits.detach().squeeze(), test_label.type(t.LongTensor))

    test_cm = test_confusion_matrix.value()
    acc = 100. * (test_cm.diagonal().sum()) / (test_cm.sum())
    FP, FN, TP, TN, FPR, FNR, TPR, P, F1 = cal_metrics(y_true, y_pred)

    print('acc', acc)
    print('FP: {FP}, FN: {FN}, TP: {TP}, TN: {TN}'.format(FP=FP, FN=FN, TP=TP, TN=TN))
    print('FPR: {FPR}, FNR: {FNR}, TPR: {TPR}, P: {P}, F1: {F1}'.format(FPR=FPR, FNR=FNR, TPR=TPR, P=P, F1=F1))

    print("test_cm:\n{test_cm}".format(
        test_cm=str(test_cm),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet', help="model to be used")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, required=True, help="Path of trained model")

    args = parser.parse_args()

    test(args)
