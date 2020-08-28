# built-in / external module
import random

# torch
import torch
from torch.utils.data import Dataset
import options


obj = type('obj', (), {})
class TimeSeriesDataset(Dataset):
    def __init__(self, options):
        self.options = options

        #xlsx_obj = pd.read_excel(self.options.xlsx_path)
        #json_obj = json.load(open(self.options.json_path))
        #pkl_obj = pickle.load(open(self.options.pkl_path, 'rb'))
        #hdf5_obj = h5py.File(self.options.hdf5_path, 'r')

        self.train_dataset = obj()
        self.train_dataset.x = None
        self.train_dataset.y = None

        self.validation_dataset = obj()
        self.validation_dataset.x = None
        self.validation_dataset.y = None

        self.test_dataset = obj()
        self.test_dataset.x = None
        self.test_dataset.y = None

        

        if options.dataset_name == 'independent-univariate-unistep':
            self.timestep = 5
            self.predictstep = 1
            self.dataset = self.spliter(list(range(0,1000)), self.timestep, self.predictstep)

            num_dataset = len(self.dataset)
            _validation = int(num_dataset*options.split_rate)
            _train = int(_validation*options.split_rate)
           
            train_dataset = torch.tensor(self.dataset[:_train]).type(torch.FloatTensor)
            validation_dataset = torch.tensor(self.dataset[_train:_validation]).type(torch.FloatTensor)
            test_dataset = torch.tensor(self.dataset[_validation:]).type(torch.FloatTensor)

            self.train_dataset.x = train_dataset[:, :self.timestep]
            self.train_dataset.y = train_dataset[:, self.timestep:]
            self.validation_dataset.x = validation_dataset[:, :self.timestep]
            self.validation_dataset.y = validation_dataset[:, self.timestep:]
            self.test_dataset.x = test_dataset[:, :self.timestep]
            self.test_dataset.y = test_dataset[:, self.timestep:]

        elif options.dataset_name == 'independent-univariate-multistep':
            self.timestep = 5
            self.predictstep = 3
            self.dataset = self.spliter(list(range(0,1000)), self.timestep, self.predictstep)

            num_dataset = len(self.dataset)
            _validation = int(num_dataset*options.split_rate)
            _train = int(validation*options.split_rate)
           
            train_dataset = torch.tensor(self.dataset[:_train]).type(torch.FloatTensor)
            validation_dataset = torch.tensor(self.dataset[_train:_validation]).type(torch.FloatTensor)
            test_dataset = torch.tensor(self.dataset[_validation:]).type(torch.FloatTensor)

            self.train_dataset.x = train_dataset[:, :self.timestep]
            self.train_dataset.y = train_dataset[:, self.timestep:]
            self.validation_dataset.x = validation_dataset[:, :self.timestep]
            self.validation_dataset.y = validation_dataset[:, self.timestep:]
            self.test_dataset.x = test_dataset[:, :self.timestep]
            self.test_dataset.y = test_dataset[:, self.timestep:]

        elif options.dataset_name == 'independent-multivariate-unistep':
            self.variate = 3
            self.timeseries = list(range(0,1000))
            self.timestep = 5
            self.predictstep = 1

            for i in range(self.variate):
                # self.dataset
                setattr(self, f'dataset{i}', self.spliter(self.timeseries, self.timestep, self.predictstep))

            num_dataset = len(self.dataset0)
            _validation = int(num_dataset*options.split_rate)
            _train = int(validation*options.split_rate)
           
            for i in range(self.variate):
            train_dataset = torch.tensor(self.dataset[:_train]).type(torch.FloatTensor)
            validation_dataset = torch.tensor(self.dataset[_train:_validation]).type(torch.FloatTensor)
            test_dataset = torch.tensor(self.dataset[_validation:]).type(torch.FloatTensor)

            self.train_dataset.x = train_dataset[:, :self.timestep]
            self.train_dataset.y = train_dataset[:, self.timestep:]
            self.validation_dataset.x = validation_dataset[:, :self.timestep]
            self.validation_dataset.y = validation_dataset[:, self.timestep:]
            self.test_dataset.x = test_dataset[:, :self.timestep]
            self.test_dataset.y = test_dataset[:, self.timestep:]

        elif options.dataset_name == 'independent-multivariate-multistep':
            self.variate = 3
            self.timeseries = list(range(0,1000))
            self.timestep = 8
            self.predictstep = 3

            for i in range(self.variate):
                # self.dataset
                setattr(self, f'dataset{i}', self.spliter(self.timeseries, self.timestep, self.predictstep))

            num_dataset = len(self.dataset0)
            _validation = int(num_dataset*options.split_rate)
            _train = int(validation*options.split_rate)
           
            for i in range(self.variate):
            train_dataset = torch.tensor(self.dataset[:_train]).type(torch.FloatTensor)
            validation_dataset = torch.tensor(self.dataset[_train:_validation]).type(torch.FloatTensor)
            test_dataset = torch.tensor(self.dataset[_validation:]).type(torch.FloatTensor)

            self.train_dataset.x = train_dataset[:, :self.timestep]
            self.train_dataset.y = train_dataset[:, self.timestep:]
            self.validation_dataset.x = validation_dataset[:, :self.timestep]
            self.validation_dataset.y = validation_dataset[:, self.timestep:]
            self.test_dataset.x = test_dataset[:, :self.timestep]
            self.test_dataset.y = test_dataset[:, self.timestep:]

        elif options.dataset_name == 'dependent-univariate-unistep':
            pass
        elif options.dataset_name == 'dependent-univariate-multistep':
            pass
        elif options.dataset_name == 'dependent-multivariate-unistep':
            pass
        elif options.dataset_name == 'dependent-multivariate-multistep':
            pass
        
        self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
        self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
        self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
        self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
        self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
        self.options.add.y_test_shape = self.test_dataset.y.size()[1:]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset.y)
        elif self.mode == 'validation':
            return len(self.validation_dataset.y)
        elif self.mode == 'test':
            return len(self.test_dataset.y)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x_item = self.train_dataset.x[idx].to(self.options.device)
            y_item = self.train_dataset.y[idx].to(self.options.device)
        elif self.mode == 'validation':
            x_item = self.validation_dataset.x[idx].to(self.options.device)
            y_item = self.validation_dataset.y[idx].to(self.options.device)
        elif self.mode == 'test':
            x_item = self.test_dataset.x[idx].to(self.options.device)
            y_item = self.test_dataset.y[idx].to(self.options.device)
        return x_item, y_item
    
    def type(self, mode='train'):
        self.mode = mode
        x_size = getattr(self.options.add, 'x_'+mode+'_shape')
        y_size = getattr(self.options.add, 'y_'+mode+'_shape')
        print(f'[DATASET][{mode.upper()}] input size : {x_size}')
        print(f'[DATASET][{mode.upper()}] target size : {y_size}')
        return self
 
    def generater(self, variate, timeseries, timestep, predictstep):
        for i in range(variate):
            # self.dataset
            setattr(self, f'dataset{i}', self.spliter(timeseries, timestep, predictstep))

        num_dataset = len(self.dataset0)
        _validation = int(num_dataset*options.split_rate)
        _train = int(validation*options.split_rate)
       
        for i in range(self.variate):
        train_dataset = torch.tensor(self.dataset[:_train]).type(torch.FloatTensor)
        validation_dataset = torch.tensor(self.dataset[_train:_validation]).type(torch.FloatTensor)
        test_dataset = torch.tensor(self.dataset[_validation:]).type(torch.FloatTensor)

        self.train_dataset.x = train_dataset[:, :self.timestep]
        self.train_dataset.y = train_dataset[:, self.timestep:]
        self.validation_dataset.x = validation_dataset[:, :self.timestep]
        self.validation_dataset.y = validation_dataset[:, self.timestep:]
        self.test_dataset.x = test_dataset[:, :self.timestep]
        self.test_dataset.y = test_dataset[:, self.timestep:]

        self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
        self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
        self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
        self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
        self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
        self.options.add.y_test_shape = self.test_dataset.y.size()[1:]


    @staticmethod
    def spliter(time_series, time_step, predict_step):
        dataset = list()

        for i in range(len(time_series)):
            end_index = i + (time_step + predict_step)
            if end_index > len(time_series)-1 : break
            
            sequence = time_series[i:end_index]
            dataset.append(sequence)
	
        random.shuffle(dataset)
        return dataset



