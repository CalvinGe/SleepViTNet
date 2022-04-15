import torch
from vit_pytorch import ViT
from losses import *
from models import *
import data
from Forward import *
import numpy as np
import random
import os


device = torch.device("cuda:0")

model = Model(image_size=(1, 3000),
            patch_size=(1, 100), 
            num_classes=5,
            dim=1024,
            depth=6,
            channels=3,
            heads=16,
            mlp_dim=2048,
            dropout=0.3, 
            emb_dropout=0.1).to(device)

# model.load_state_dict(torch.load("./weights/Wres120_depth6_1.pth")['ViT'])

BATCH_SIZE = 128
train_loader, test_loader = data.data_preparation(BATCH_SIZE=BATCH_SIZE, num_id=197)

if not os.path.exists('weights'):
    os.mkdir('weights')

filename = './weights/Wres120_depth6_gradLr_'
result_file = "Wres120_15_35_2_2_5_depth6_gradLr.txt"

optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-4}])

def adjust_learning_rate_list(optimizer, epoch):
    lr_set_list = [[1,1e-4],[2,5e-5],[3,1e-5],[4,1e-6],[5,1e-7]] 
    lr_list = []
    for i in lr_set_list:
        for j in range(i[0]):
            lr_list.append(i[1])
    for param_group in optimizer.param_groups:
        if epoch < len(lr_list)-1:
            param_group['lr'] = lr_list[epoch]
        else:
            param_group['lr'] = lr_list[-1]

# optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': 1e-3, 'momentum': 0.9}])

criterion1 = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.5, 3.5, 2, 2, 5]).to(device))
# criterion3 = DiceLoss().to(device)
criterion2 = SupConLoss(temperature=0.1).to(device)



baseline = 1e7
for epoch in range(6):
    with open(result_file, 'a+') as f:
        f.write(str(epoch) + '\n')

    adjust_learning_rate_list(optimizer, epoch)
    train(optimizer,criterion1,criterion2, train_loader, model, device)

    test_loss = test(criterion1, criterion2, test_loader, model, device)
    if test_loss < baseline:
        baseline = test_loss
        state = {'ViT': model.state_dict()}
        torch.save(state, filename + str(epoch) + '.pth')
