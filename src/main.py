import torch
from torch import nn

from dataset import KeyPointDataset
from torch.utils.data import DataLoader

from models import KeyPointLearner
from conf import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(dataloader, learner, criterion, optimizer):
    for i_epoch in range(EPOCH):
        for sub, (label, keypoints, keypoints_m) in enumerate(dataloader):
            label = label.to(device)
            keypoints = keypoints.to(device).float()
            keypoints_m = keypoints_m.to(device).float()
            pred = learner(keypoints, keypoints_m)
            # print(pred.size(), label.size())
            loss_v = criterion(input=pred, target=label)
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
            if sub % 50 == 0:
                print(f'loss = {loss_v}')
                pred_res = pred.argmax(dim=1)
                correct = (pred_res == label).sum().item()
                # print(f'correct = {correct * 1. / 16 * 100}%')


def test(dataloader, learner):
    correct = 0
    total = 0
    for sub, (label, keypoints, keypoints_m) in enumerate(dataloader):
        label = label.to(device)
        keypoints = keypoints.to(device).float()
        keypoints_m = keypoints_m.to(device).float()
        pred = learner(keypoints, keypoints_m)
        pred_res = pred.argmax(dim=1)
        correct += (pred_res == label).sum().item()
        total += pred.size()[0]
    print(f'correct rate = {correct * 1.0 / total * 100}%')


if __name__ == '__main__':
    train_dataloader = DataLoader(
        KeyPointDataset('../test/resource/res'),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        KeyPointDataset('../test/resource/res2'),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    learner = KeyPointLearner().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.002)

    train(train_dataloader, learner, criterion, optimizer)
    test(val_dataloader, learner)
