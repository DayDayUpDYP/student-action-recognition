import torch
from torch import nn

from dataset import KeyPointDataset
from torch.utils.data import DataLoader

from models import KeyPointLearner, KeyPointLearnerGAT

from conf import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def save_model(path, model: nn.Module):
    torch.save(model.state_dict(), path)


def load_model(path, model: nn.Module):
    model.load_state_dict(torch.load(path))


def train(dataloader, learner, criterion, optimizer, with_metric=False):
    for i_epoch in range(EPOCH):
        for sub, (label, keypoints_pm, keypoints_m) in enumerate(dataloader):
            label = label.to(device)
            keypoints_pm = keypoints_pm.to(device).float()
            keypoints_m = keypoints_m.to(device).float()
            pred = learner(keypoints_pm, keypoints_m)
            # print(keypoints.type(), keypoints_m.type())
            # print(pred.size(), label.size())
            loss_v = criterion(input=pred, target=label)
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
            if (sub * i_epoch + 1) % 100 == 0 and with_metric:
                print(f'epoch = {i_epoch}, loss = {loss_v}')


def test(dataloader, learner):
    learner.eval()
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
    return correct * 1. / total


if __name__ == '__main__':
    train_dataloader = DataLoader(
        KeyPointDataset('../test/resource/output', keypoint_num=26),
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        KeyPointDataset('../test/resource/val', keypoint_num=26),
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    total_rate = 0

    learner = KeyPointLearnerGAT(1, 5).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(learner.parameters(), lr=0.0005, weight_decay=1.e-4)

    train(train_dataloader, learner, criterion, optimizer, with_metric=True)

    total_rate += test(val_dataloader, learner)

    save_model('../test/resource/model.pkl', learner)

    # for ins_num in range(1, 100):
    #     total_rate = 0.
    #
    #     for _ in range(CNT):
    #         print(f'CNT:{_}')
    #
    #         learner = KeyPointLearner(intensify_num=ins_num * 0.1).to(device)
    #
    #         criterion = nn.CrossEntropyLoss().to(device)
    #         optimizer = torch.optim.Adam(learner.parameters(), lr=0.0005, weight_decay=1.e-4)
    #
    #         train(train_dataloader, learner, criterion, optimizer, with_metric=True)
    #
    #         total_rate += test(val_dataloader, learner)
    #
    #         save_model('../test/resource/model.pkl', learner)
    #
    #     print(f'intensify_num: {ins_num}, total correct rate: {total_rate / CNT * 100}%')

    # load_model('../test/resource/model.pkl', learner)
