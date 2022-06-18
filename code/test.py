import torch
import torch.nn as nn

from MobileNet import *
from UNSWGRAYDATASET import *
from UNSWORIGINDATASET import *
from UNSWBINARYDATASET import *

from torch.utils.data import DataLoader
import numpy as np 

def test(model, test_loader, pt_file, data_len):
    print("Model weight load")
    model.load_state_dict(torch.load('./ptfiles/' + pt_file))
    model = model.cuda()
    print("Model weight load Complete")
    
    model.eval()
    
    correct = 0

    print("Testing.....")
    Catertory = []
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
        Category = Category + output.tolist()
        output = output.cpu()

        for i in range(len(output)):

            if(output[i] == target[i]):
                correct += 1
                tmp_correct += 1
        
        # print("Current acc => ", tmp_correct / len(output))
    total_acc = correct/len(data_len)
    print("==========================================")
    print(pt_file, " 's Total Acc =>", total_acc)
    print("==========================================")

    Id = list(range(0, data_len))
    samples = {
       'Id': Id,
       'Category': Category 
    }

    df = pd.DataFrame(samples, columns=['Id', 'Category'])

    df.to_csv(f'./{pt_file}_result.csv', index=False)
    return total_acc

if __name__ == "__main__":
    print("Model load")
    model = MobileNetV1(ch_in=1, n_classes=2)
    print("Model load Complete")

    print("Making Dataset.... ")
    binary_data = MyDataSet_TEST_donotmix()
    print("Binary file end..")
    gray_data = UNSWGRAYDATASETTEST()
    print("Gray file end..")
    origin_data = UNSWORIGINDATASETTEST()
    print("Original file end..")
    print("Making Dataset complete! ")

    print("Making Data Loaders")
    binary_loader = DataLoader(binary_data, batch_size = 64, shuffle=False)
    print("Binary data on")
    gray_loader = DataLoader(gray_data, batch_size = 64, shuffle=False)
    print("Gray data on")
    origin_loader = DataLoader(origin_data, batch_size = 64, shuffle=False)
    print("Origin data on")
    
    binary_ACC_List = []
    gray_ACC_List = []
    origin_ACC_List = []

    binary_pt_list = ['binarytraining_9.pt',           'binarytraining_19.pt',           'binarytraining_29.pt',
           'binarytraining_39.pt',            'binarytraining_49.pt',            'binarytraining_59.pt',
           'binarytraining_69.pt',            'binarytraining_79.pt',            'binarytraining_89.pt',
           'binarytraining_99.pt',            'binarytraining_109.pt',           'binarytraining_119.pt',
           'binarytraining_129.pt',           'binarytraining_139.pt',           'binarytraining_149.pt',
           'binarytraining_159.pt',           'binarytraining_169.pt',           'binarytraining_179.pt',
           'binarytraining_189.pt',           'binarytraining_199.pt',           'binarytraining_209.pt',
           'binarytraining_219.pt',           'binarytraining_229.pt',           'binarytraining_239.pt',
           'binarytraining_249.pt',           'binarytraining_259.pt',           'binarytraining_269.pt',
           'binarytraining_279.pt',           'binarytraining_289.pt',           'binarytraining_299.pt'
          ]

    gray_pt_list = ['graytraining_9.pt',           'graytraining_19.pt',           'graytraining_29.pt',
           'graytraining_39.pt',            'graytraining_49.pt',            'graytraining_59.pt',
           'graytraining_69.pt',            'graytraining_79.pt',            'graytraining_89.pt',
           'graytraining_99.pt',            'graytraining_109.pt',           'graytraining_119.pt',
           'graytraining_129.pt',           'graytraining_139.pt',           'graytraining_149.pt',
           'graytraining_159.pt',           'graytraining_169.pt',           'graytraining_179.pt',
           'graytraining_189.pt',           'graytraining_199.pt',           'graytraining_209.pt',
           'graytraining_219.pt',           'graytraining_229.pt',           'graytraining_239.pt',
           'graytraining_249.pt',           'graytraining_259.pt',           'graytraining_269.pt',
           'graytraining_279.pt',           'graytraining_289.pt',           'graytraining_299.pt'
          ]

    original_pt_list = [
           'origintraining_9.pt',             'origintraining_19.pt',            'origintraining_29.pt',
           'origintraining_39.pt',            'origintraining_49.pt',            'origintraining_59.pt',
           'origintraining_69.pt',            'origintraining_79.pt',            'origintraining_89.pt',
           'origintraining_99.pt',            'origintraining_109.pt',           'origintraining_119.pt',
           'origintraining_129.pt',           'origintraining_139.pt',           'origintraining_149.pt',
           'origintraining_159.pt',           'origintraining_169.pt',           'origintraining_179.pt',
           'origintraining_189.pt',           'origintraining_199.pt',           'origintraining_209.pt',
           'origintraining_219.pt',           'origintraining_229.pt',           'origintraining_239.pt',
           'origintraining_249.pt',           'origintraining_259.pt',           'origintraining_269.pt',
           'origintraining_279.pt',           'origintraining_289.pt',           'origintraining_299.pt'
          ]

    for pt_file in binary_pt_list:
        binary_ACC_List.append(test(model, binary_loader, pt_file))
    
    for pt_file in gray_pt_list:
        gray_ACC_List.append(test(model, gray_loader, pt_file))

    for pt_file in original_pt_list:
        origin_ACC_List.append(test(model, origin_loader, pt_file))

    binary_ACC_List = np.array(binary_ACC_List)
    np.save('./20220618_binary_ACC_List', binary_ACC_List)

    gray_ACC_List = np.array(gray_ACC_List)
    np.save('./20220618_gray_ACC_List', gray_ACC_List)

    original_pt_list = np.array(original_pt_list)
    np.save('./20220618_original_pt_list', original_pt_list)

    print("Finished!")
