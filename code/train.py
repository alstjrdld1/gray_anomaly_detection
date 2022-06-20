import time
import numpy as np 

import torch
import torch.nn as nn 

from torch.utils.data import Dataset, DataLoader

from MobileNet import *
from UNSWBINARYDATASET import *
from UNSWGRAYDATASET import *
from UNSWORIGINDATASET import *

WEIGHTDECAY = 0.1e-4
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
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))
        
        # train for one epoch 
        start_time = time.time()
#         last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [75,125], gamma=0.1)
        avg_loss = train(train_loader, epoch, model, optimizer, criterion)
        avg_loss_list.append(avg_loss)
        scheduler.step()
        elapsed_time = time.time() - start_time 
        print('==> {:.2f} seconds to  train this epoch \n'.format(
                elapsed_time))
        
        # learning rate scheduling 
        if(epoch % 10 == 9):
            torch.save(model.state_dict(), f'./ptfiles/{save_name}_{epoch}.pt')

    avg_loss_list = np.array(avg_loss_list)
    np.save(f'./{save_name}_loss', avg_loss_list)

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
        loss_avg = sum(batch_loss) / len(batch_loss)
    return loss_avg
#############################################################################
#############################################################################

if __name__ == "__main__":
    print("Making Dataset.... ")
    # binary_data = UNSWBINARYDATASET()
    # print("Binary file end..")
    # gray_data = UNSWGRAYDATASET()
    # print("Gray file end..")
    origin_data = UNSWORIGINDATASET()
    print("Original file end..")
    print("Making Dataset complete! ")

    print("Making Data Loaders")
    # binary_loader = DataLoader(binary_data, batch_size = BATCHSIZE, shuffle=True)
    # print("Binary data on")

    # gray_loader = DataLoader(gray_data, batch_size = BATCHSIZE, shuffle=True)
    # print("Gray data on")
    origin_loader = DataLoader(origin_data, batch_size = BATCHSIZE, shuffle=True)
    print("Origin data on")

    model = MobileNetV1(ch_in=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr = LR,
                               momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                               nesterov=True)
    

    criterion = torch.nn.CrossEntropyLoss()

    print("Binary Model training start")
    # main(model=model1, train_loader=binary_loader, optimizer=optimizer1, criterion=criterion, save_name="binarytraining")
    # print("MobileNet with BINARYDATASET CLEAR!")

    # print("gray Model training start")
    # main(model=model, train_loader=gray_loader, optimizer=optimizer, criterion=criterion, save_name="graytraining")
    # print("MobileNet with GRAYDATASET CLEAR!")

    print("Origin Model training start")
    main(model=model, train_loader=origin_loader, optimizer=optimizer, criterion=criterion, save_name="origintraining")
    # print("MobileNet with ORIGINDATASET CLEAR!")
