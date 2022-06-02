
from torch.utils.data import Dataset
from my_utils import *

class UNSWGRAYDATASET(Dataset):
    def __init__(self):
        Normal_data = pd.read_csv('../Normalized_normal_train.csv', index_col=False)
        Anomaly_data = pd.read_csv('../Normalized_anomal_train.csv', index_col=False)
        
        normal_packets = Normal_data.values
        anomaly_packets = Anomaly_data.values

        # Make normal patch and anomaly patch
        print("Making gray rows....")
        normal_rows = []
        for packet in normal_packets:
            normal_rows.append(make_gray_row(packet, 64))
        
        anomaly_rows = []
        for packet in anomaly_packets:
            anomaly_rows.append(make_gray_row(packet, 64))
        
        # Make features
        self.x_train = []
        self.y_train = []

        print("Making gray patches....")
        for i in range(len(normal_rows) - 64):
            self.x_train.append(make_gray_patch(normal_rows[i:i+64]))
            self.y_train.append(0)
        
        for i in range(len(anomaly_rows) - 64):
            self.x_train.append(make_gray_patch(anomaly_rows[i:i+64]))
            self.y_train.append(1)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class UNSWGRAYDATASETTEST(Dataset):
    def __init__(self):
        Normal_data = pd.read_csv('../Normalized_normal_test.csv', index_col=False)
        Anomaly_data = pd.read_csv('../Normalized_anomal_test.csv', index_col=False)
        
        normal_packets = Normal_data.values
        anomaly_packets = Anomaly_data.values

        # Make normal patch and anomaly patch
        print("Making gray rows....")
        normal_rows = []
        for packet in normal_packets:
            normal_rows.append(make_gray_row(packet, 64))
        
        anomaly_rows = []
        for packet in anomaly_packets:
            anomaly_rows.append(make_gray_row(packet, 64))
        
        # Make features
        self.x_test = []
        self.y_test = []

        print("Making gray patches....")
        for i in range(len(normal_rows) - 64):
            self.x_test.append(make_gray_patch(normal_rows[i:i+64]))
            self.y_test.append(0)
        
        for i in range(len(anomaly_rows) - 64):
            self.x_test.append(make_gray_patch(anomaly_rows[i:i+64]))
            self.y_test.append(1)
  
    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        return self.x_test[idx], self.y_test[idx]