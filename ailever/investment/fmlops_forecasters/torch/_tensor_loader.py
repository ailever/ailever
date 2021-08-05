import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class InvestmentDataset(Dataset):
    def __init__(self, training_info):
        ticker = training_info['ticker']
        df = pd.read_csv(f'temp/{ticker}.csv')[['Date', 'Close',  'Volume_PCT', 'VIX_Rate', 'VIX_Close_PCT', 'US10YT_Rate', 'Price_Rate']]
        df.Date = pd.to_datetime(df.Date.astype('str'))
        df = df.set_index('Date')

        self.device = training_info['device']
        self.packet_size = 365
        self.predict_range = 100
        self.train_range = self.packet_size - self.predict_range
        self.feature_weights = dict()
        self.feature_weights['Volume'] = 0.01

        self.frame = df
        self.frame_train = df.iloc[:1500]
        self.frame_test = df.iloc[1500:]
        self.frame_last_packet = self.frame.iloc[-self.packet_size:]
        self.tensor_train = torch.from_numpy(self.frame_train.values)
        self.tensor_test = torch.from_numpy(self.frame_test.values)
        self.tensor_last_packet = torch.from_numpy(self.frame_last_packet.values)


    def __len__(self):
        if self.mode == 'train':
            return self.tensor_train.size()[0] - self.packet_size
        elif self.mode == 'test':
            return self.tensor_test.size()[0] - self.packet_size

    def __getitem__(self, idx):
        S = _Scaler()
        if self.mode == 'train':
            time_series = S.standard(self.tensor_train[idx:idx+self.packet_size])
        elif self.mode == 'test':
            time_series = S.standard(self.tensor_test[idx:idx+self.packet_size])

        x_item = time_series[:self.train_range].to(self.device)
        y_item = time_series[self.train_range:].to(self.device)
        return x_item, y_item

    def type(self, mode='train'):
        self.mode = mode
        self.frame = getattr(self, f'frame_{mode}')
        return self

dataset = InvestmentDataset()
train_dataloader = DataLoader(dataset.type('train'), batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(dataset.type('test'), batch_size=batch_size, shuffle=True, drop_last=False)


