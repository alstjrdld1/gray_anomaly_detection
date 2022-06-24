
from torch.utils.data import Dataset
from my_utils import *

class UNSWGRAYDATASET(Dataset):
    def __init__(self):
        # Normal_data = pd.read_csv('../Normalized_normal_train.csv', index_col=False)
        # Anomaly_data = pd.read_csv('../Normalized_anomal_train.csv', index_col=False)
        
        Normal_data = pd.read_csv('../UNSW_NB15_NORMAL.csv', index_col=False)
        Anomaly_data = pd.read_csv('../UNSW_NB15_ANOMALY.csv', index_col=False)
        
        normal_packets = Normal_data.drop(['attack_cat', 'label'], axis=1).values
        anomaly_packets = Anomaly_data.drop(['attack_cat', 'label'], axis=1).values

        # normal_packets = Normal_data.values
        # anomaly_packets = Anomaly_data.values

        # Make normal patch and anomaly patch
        print("Making gray rows....")
        normal_rows = []
        for packet in normal_packets:
            normal_rows.append(make_row(packet, 64))
        
        anomaly_rows = []
        for packet in anomaly_packets:
            anomaly_rows.append(make_row(packet, 64))
        
        # Make features
        self.x_train = []
        self.y_train = []

        print("Making gray patches....")
        for idx, _ in enumerate(normal_rows):
            if( (idx + 64) > len(normal_rows)):
                break
            
            patch = make_gray_patch(normal_rows[idx: idx+64])            
            patch *= 255/patch.max()
            self.x_train.append(patch)
            self.y_train.append(0)
        
        print("Making Gray Anomaly patches....")
        for idx, _ in enumerate(anomaly_rows):
            if( (idx + 64) > len(anomaly_rows)):
                break
                
            patch = make_gray_patch(anomaly_rows[idx: idx+64])
            patch *= 255/patch.max()

            self.x_train.append(patch)
            self.y_train.append(1)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class UNSWGRAYDATASETTEST(Dataset):
    def __init__(self):
        Normal_data = pd.read_csv('../UNSW_NB15_TEST_NORMAL.csv', index_col=False)
        Anomaly_data = pd.read_csv('../UNSW_NB15_TEST_ANOMALY.csv', index_col=False)
        
        normal_packets = Normal_data.values
        anomaly_packets = Anomaly_data.values

        # Make normal patch and anomaly patch
        print("Making gray rows....")
        normal_rows = []
        for packet in normal_packets:
            normal_rows.append(make_row(packet, 64))
        
        anomaly_rows = []
        for packet in anomaly_packets:
            anomaly_rows.append(make_row(packet, 64))
        
        # Make features
        self.x_test = []
        self.y_test = []

        print("Making gray patches....")
        for idx, _ in enumerate(normal_rows):
            if( (idx + 64) > len(normal_rows)):
                break
            
            patch = make_gray_patch(normal_rows[idx: idx+64])            
            patch *= 255/patch.max()
            self.x_test.append(patch)
            self.y_test.append(0)
        
        print("Making Gray Anomaly patches....")
        for idx, _ in enumerate(anomaly_rows):
            if( (idx + 64) > len(anomaly_rows)):
                break
                
            patch = make_gray_patch(anomaly_rows[idx: idx+64])
            patch *= 255/patch.max()

            self.x_test.append(patch)
            self.y_test.append(1)
        
        answer = np.array(self.y_test)
        answer = pd.DataFrame(answer)
        answer.to_csv('GRAYANSWER.csv', index=False)

    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        return self.x_test[idx], self.y_test[idx]