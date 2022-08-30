import time
import numpy as np 

import torch
import torch.nn as nn 

from torch.utils.data import Dataset, DataLoader

from MobileNet import *
from UNSWBINARYDATASET import *
from UNSWGRAYDATASET import *
from UNSWORIGINDATASET import *

WEIGHTDECAY = 1e-4
MOMENTUM = 0.9
BATCHSIZE = 64
LR = 0.1
EPOCHS = 150
#############################################################################
################################ TRAIN CODE #################################
def main(model, train_loader, optimizer, criterion, save_name):
    model = model.cuda()
    criterion = criterion.cuda()
    
    # parameter of our model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    
    last_top1_acc = 0
    avg_loss_list = []
    acc_list = []
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))
        
        # train for one epoch 
        start_time = time.time()
#         last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [75,125], gamma=0.1)
        avg_loss, avg_acc = train(train_loader, epoch, model, optimizer, criterion)
        
        avg_loss_list.append(avg_loss)
        acc_list.append(avg_acc)

        scheduler.step()
        elapsed_time = time.time() - start_time 
        print('==> {:.2f} seconds to  train this epoch \n'.format(
                elapsed_time))
        
        # learning rate scheduling 
        if(epoch % 10 == 9):
            torch.save(model.state_dict(), f'./ptfiles/{save_name}_{epoch}.pt')

    avg_loss_list = np.array(avg_loss_list)
    np.save(f'./{save_name}_loss', avg_loss_list)
    acc_list = np.array(acc_list)
    np.save(f'./{save_name}_acc', acc_list)


def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time,  data_time, losses, 
                             top1, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode 
    model.train()
    end = time.time()
    
    batch_loss = []
    trainAcc = []
    total = 0 
    correct = 0
    best_acc = 0
    PRINTFREQ = 20
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time 
        data_time.update(time.time() - end)
        # print("input length : ", len(input))

        input = np.array(input)
        target = np.array(target)
                
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target)

        input = input.unsqueeze(1)
                
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        target = target.squeeze(-1)
        
        # compute ouput 
        output = model(input)

        loss = criterion(output, target)
        _, predicted = output.max(1)
        total += target.size(0)
        
        correct += predicted.eq(target).sum().item()
        acc1 = correct/total

        # measure accuracy and record loss, accuracy         
        losses.update(loss.item(), input.size(0))
        
        # compute gradient and do SGD step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()
        if i % PRINTFREQ == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, i * len(input), len(train_loader.dataset, ),
                       100. * i / len(train_loader), loss.item(), 100. * correct / total))
        batch_loss.append(loss.item())

        trainAcc.append(100. * correct / total)
    loss_avg = sum(batch_loss) / len(batch_loss)
    acc_avg = sum(trainAcc) / len(trainAcc)
    return loss_avg, acc_avg
#############################################################################
#############################################################################

class AvalancheDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('./abnormals/total.csv', index_col=False)
        # data = pd.read_csv('./abnormals/UDP_Flooding.csv', index_col=False)
        data = data.drop(['No.'], axis=1).values
        patches = []
        for dat in data:
            patches.append(make_patch(dat, (32, 32)))
        
        self.x_test = []
        self.y_test = []

        for idx,_ in enumerate(patches):
            pf = PacketFeature((224, 224))
            if( (idx + 49) > len(patches)):
                break
        
            for count in range(49):
                pf.append(patches[idx+count])

            self.y_test.append(1)
            self.x_test.append(pf.frame)
    
    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        return self.x_test[idx], self.y_test[idx]

if __name__ == "__main__":
    print("Making Dataset.... ")
    # train_data = UNSWBINARYDATASET()
    # train_data = UNSWORIGINDATASET()
    # train_data = UNSWGRAYDATASET()
    train_data = AvalancheDataset()
    print("Binary file end..")
    print("Making Dataset complete! ")

    print("Making Data Loaders")
    train_loader = DataLoader(train_data, batch_size = BATCHSIZE, shuffle=True)
    print("Binary data on")

    model = MobileNetV1(ch_in=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr = LR,
                               momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                               nesterov=True)
    

    criterion = torch.nn.CrossEntropyLoss()

    print("Binary Model training start")
    main(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, save_name="binarytraining_0830_avalanche")
    print("MobileNet with BINARYDATASET CLEAR!")
