
from operator import index
from torch.utils.data import Dataset
from my_utils import *


#############################################################################
######################### Dataset Preprocessing #############################
class UNSWBINARYDATASET(Dataset):
    def __init__(self):
        Normal_data = pd.read_csv('../UNSW_NB15_NORMAL.csv', index_col=False)
        Anomaly_data = pd.read_csv('../UNSW_NB15_ANOMALY.csv', index_col=False)
        
        normal_packets = Normal_data.drop(['attack_cat', 'label'], axis=1).values
        anomaly_packets = Anomaly_data.drop(['attack_cat', 'label'], axis=1).values

        # Make normal patch and anomaly patch
        normal_patches = []
        for packet in normal_packets:
            normal_patches.append(make_patch(packet, (32,32)))
        
        anomaly_patches = []
        for packet in anomaly_packets:
            anomaly_patches.append(make_patch(packet, (32,32)))
        
        # Make features
        self.x_train = []
        self.y_train = []

        print("Appending Normal Data....")
        for idx, _ in enumerate(normal_patches):
            pf = PacketFeature((224,224))
            if( (idx + 49) > len(normal_patches)):
                break
                
            for count in range(49):
                pf.append(normal_patches[idx+count])
            
            self.y_train.append(0)
            self.x_train.append(pf.frame)
        print("Appending Normal Data End! ")

        print("Appending Anomaly Data....")
        for idx, _ in enumerate(anomaly_patches):
            pf = PacketFeature((224,224))
            if( (idx + 49) > len(anomaly_patches)):
                break
                
            for count in range(49):
                pf.append(anomaly_patches[idx+count])
            
            self.y_train.append(1)
            self.x_train.append(pf.frame)        
        print("Appending Anomaly Data End! ")

  
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
#############################################################################
#############################################################################

class UNSWBINARYDATASETTEST(Dataset):
    def __init__(self):
        Normal_data = pd.read_csv('../UNSW_NB15_TEST_NORMAL.csv', index_col=False)
        Anomaly_data = pd.read_csv('../UNSW_NB15_TEST_ANOMALY.csv', index_col=False)
        
        normal_packets = Normal_data.drop(['attack_cat', 'label'], axis=1).values
        anomaly_packets = Anomaly_data.drop(['attack_cat', 'label'], axis=1).values

        # Make normal patch and anomaly patch
        normal_patches = []
        for packet in normal_packets:
            normal_patches.append(make_patch(packet, (32,32)))
        
        anomaly_patches = []
        for packet in anomaly_packets:
            anomaly_patches.append(make_patch(packet, (32,32)))
        
        # Make features
        self.x_train = []
        self.y_train = []

        print("Appending Normal Data....")
        for idx, _ in enumerate(normal_patches):
            pf = PacketFeature((224,224))
            if( (idx + 49) > len(normal_patches)):
                break
                
            for count in range(49):
                pf.append(normal_patches[idx+count])
            
            self.y_train.append(0)
            self.x_train.append(pf.frame)
        print("Appending Normal Data End! ")

        print("Appending Anomaly Data....")
        for idx, _ in enumerate(anomaly_patches):
            pf = PacketFeature((224,224))
            if( (idx + 49) > len(anomaly_patches)):
                break
                
            for count in range(49):
                pf.append(anomaly_patches[idx+count])
            
            self.y_train.append(1)
            self.x_train.append(pf.frame)        
        print("Appending Anomaly Data End! ")

  
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
