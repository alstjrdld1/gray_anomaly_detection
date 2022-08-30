import sys 
import torch
import torch.nn as nn

from MobileNet import *
from UNSWBINARYDATASET import *
from UNSWORIGINDATASET import *
from UNSWGRAYDATASET import *

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def test(model, test_loader, ptfile):
    print("Model weight load")
    model.load_state_dict(torch.load('./ptfiles/'+ptfile))
    model = model.cuda()
    print("Model weight load Complete")
    model.eval()
    
    correct = 0
    Category = []
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
        Category = Category + output.tolist()

        for i in range(len(output)):

            if(output[i] == target[i]):
                correct += 1
                tmp_correct += 1
        
        print("Current acc => ", tmp_correct / len(output))
    acc = correct/len(test_data)
    print("==========================================")
    print("Total Acc =>", acc)
    print("==========================================")

    df = pd.DataFrame(Category, columns=['Category'])
    df.to_csv('./results/'+ptfile+'.csv', index=False)
    print(ptfile, "done!")

    return acc

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
    pt_file = sys.argv[1]
    print("Model load")
    model = MobileNetV1(ch_in=1, n_classes=2)
    print("Model load Complete")

    print("loading test_data")
    # test_data = MyDataSet_TEST()
    test_data = AvalancheDataset()
    # test_data = UNSWORIGINDATASETTEST()
    # test_data = UNSWGRAYDATASETTEST()
    print("loading test_data complete")

    test_loader = DataLoader(test_data, batch_size = 64, shuffle=False)
    
    # pt_files = [
    #     'binarytraining_9.pt',  'binarytraining_19.pt',
    #     'binarytraining_29.pt', 'binarytraining_39.pt',
    #     'binarytraining_49.pt', 'binarytraining_59.pt',
    #     'binarytraining_69.pt', 'binarytraining_79.pt',
    #     'binarytraining_89.pt', 'binarytraining_99.pt',
    #     'binarytraining_109.pt','binarytraining_119.pt',
    #     'binarytraining_129.pt','binarytraining_139.pt',
    #     'binarytraining_149.pt'
    # ]
    # pt_files = [
    #     'graytraining_9.pt',    'graytraining_19.pt',
    #     'graytraining_29.pt',   'graytraining_39.pt',
    #     'graytraining_49.pt',   'graytraining_59.pt',
    #     'graytraining_69.pt',   'graytraining_79.pt',
    #     'graytraining_89.pt',   'graytraining_99.pt',
    #     'graytraining_109.pt',  'graytraining_119.pt',
    #     'graytraining_129.pt',  'graytraining_139.pt',
    #     'graytraining_149.pt'
    # ]
    # pt_files = [
    #     'originaltraining_9.pt',    'originaltraining_19.pt',
    #     'originaltraining_29.pt',   'originaltraining_39.pt',
    #     'originaltraining_49.pt',   'originaltraining_59.pt',
    #     'originaltraining_69.pt',   'originaltraining_79.pt',
    #     'originaltraining_89.pt',   'originaltraining_99.pt',
    #     'originaltraining_109.pt',  'originaltraining_119.pt',
    #     'originaltraining_129.pt',  'originaltraining_139.pt',
    #     'originaltraining_149.pt'
    # ]
    test_acc_list = []
    test_acc_list.append(test(model, test_loader, pt_file))

    # for ptfile in pt_files:
    #     test_acc_list.append(test(model, test_loader, ptfile))
    
    test_acc_list = np.array(test_acc_list)
    np.save("origin_test_acc_list", test_acc_list)
 

