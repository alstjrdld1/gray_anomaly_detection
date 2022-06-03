import torch
import torch.nn as nn

from MobileNet import *
from UNSWGRAYDATASET import *

from torch.utils.data import DataLoader
import numpy as np 

def test(model, test_loader, pt_file):
    print("Model weight load")
    model.load_state_dict(torch.load('./ptfiles/' + pt_file))
    model = model.cuda()
    print("Model weight load Complete")
    
    model.eval()
    
    correct = 0

    print("Testing.....")
    for idx, (input, target) in enumerate(test_loader):
        input = np.array(input)
        input = torch.tensor(input, dtype=torch.float32)

        input = input.unsqueeze(1)
        input = input.float()
        input = input.cuda()
        
        output = model(input)
        # print("Output type : ", type(output))
        # print("Target type : ", type(target))

        # print("Output[0] type : ", type(output[0]))
        # print("Target[0] type : ", type(target[0]))
        tmp_correct= 0
        # output = output.cpu().detach().numpy()
        output = torch.argmax(output, dim=1)
        output = output.cpu()

        for i in range(len(output)):

            if(output[i] == target[i]):
                correct += 1
                tmp_correct += 1
        
        # print("Current acc => ", tmp_correct / len(output))
    total_acc = correct/len(test_data)
    print("==========================================")
    print(pt_file, " 's Total Acc =>", total_acc)
    print("==========================================")
    return total_acc

if __name__ == "__main__":
    print("Model load")
    model = MobileNetV1(ch_in=1, n_classes=2)
    print("Model load Complete")
    
    print("loading test_data")
    # test_data = MyDataSet_TEST()
    test_data = UNSWGRAYDATASETTEST()
    print("loading test_data complete")
    
    test_loader = DataLoader(test_data, batch_size = 64, shuffle=False)
    ACC_List = []

    pt_list = ['20220603_9.pt',           '20220603_19.pt',           '20220603_29.pt',
           '20220603_39.pt',           '20220603_49.pt',           '20220603_59.pt',
           '20220603_69.pt',           '20220603_79.pt',           '20220603_89.pt',
           '20220603_99.pt',           '20220603_109.pt',           '20220603_119.pt',
           '20220603_129.pt',           '20220603_139.pt',           '20220603_149.pt',
           '20220603_159.pt',           '20220603_169.pt',           '20220603_179.pt',
           '20220603_189.pt',           '20220603_199.pt',           '20220603_209.pt',
           '20220603_219.pt',           '20220603_229.pt',           '20220603_239.pt',
           '20220603_249.pt',           '20220603_259.pt',           '20220603_269.pt',
           '20220603_279.pt',           '20220603_289.pt',           '20220603_299.pt'
          ]
    
    for pt_file in pt_list:
        ACC_List.append(test(model, test_loader, pt_file))
    
    ACC_List = np.array(ACC_List)
    np.save('./20220603_acclist', ACC_List)

    print("Finished!")
