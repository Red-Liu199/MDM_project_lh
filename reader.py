import numpy as np
from config import global_config as cfg
class traffic_reader():
    
    def __init__(self):
        if cfg.year=='all': # we use all the accident data from 2014-2021
            all_data=[]
            for year in range(2015, 2022):
                all_data.append(np.load('data/accident_{}.npy'.format(year)).astype(float))
            all_data=np.concatenate(all_data, axis=0)
            if cfg.granularity=='day':
                new_data = []
                # total day: 2557
                for i in range(2557):
                    new_data.append(all_data[i*24:(i+1)*24, :, :].sum(0))
                all_data=np.stack(new_data, axis=0)
        elif cfg.year=='2015': # we additionally use green and yellow taxi data from Jan to June in 2015
            all_data=np.load('data/accident_{}.npy'.format(cfg.year)).astype(float) # S_all, H, W
            green_taxi=np.load('data/green_taxi_traffic_2015.npy').astype(float)
            yellow_taxi=np.load('data/yellow_taxi_traffic_2015.npy').astype(float)
            if cfg.granularity=='day':
                new_data, new_green, new_yellow = [], [], []
                # 181 days between Jan and June
                for i in range(181):
                    new_data.append(all_data[i*24:(i+1)*24, :, :].sum(0))
                    new_green.append(green_taxi[i*24:(i+1)*24, :, :].sum(0))
                    new_yellow.append(yellow_taxi[i*24:(i+1)*24, :, :].sum(0))
                all_data=np.stack(new_data, axis=0)
                green_taxi=np.stack(new_green, axis=0)
                yellow_taxi=np.stack(new_yellow, axis=0)
            elif cfg.granularity=='hour':
                all_data=all_data[:181*24, :, :]
                green_taxi=green_taxi[:181*24, :, :]
                yellow_taxi=yellow_taxi[:181*24, :, :]
            # normalization
            green_taxi/=np.max(green_taxi)
            yellow_taxi/=np.max(yellow_taxi)
        elif cfg.year=='2014': # we use uber, green and yellow taxi data from April to Sept in 2014
            all_data=np.load('data/accident_2014.npy'.format(cfg.year)).astype(float) # S_all, H, W
            green_taxi=np.load('data/green_taxi_traffic_2014.npy').astype(float)
            yellow_taxi=np.load('data/yellow_taxi_traffic_2014.npy').astype(float)
            uber_data=np.load('data/uber_traffic_2014.npy').astype(float)
            if cfg.granularity=='day':
                new_data, new_green, new_yellow, new_uber = [], [], [], []
                for j in range(184):
                    # there are 90 days before April and 184 days between April and September
                    i=j+90
                    new_data.append(all_data[i*24:(i+1)*24, :, :].sum(0))
                    new_green.append(green_taxi[i*24:(i+1)*24, :, :].sum(0))
                    new_yellow.append(yellow_taxi[i*24:(i+1)*24, :, :].sum(0))
                    new_uber.append(uber_data[j*24:(j+1)*24, :, :].sum(0))
                all_data=np.stack(new_data, axis=0)
                green_taxi=np.stack(new_green, axis=0)
                yellow_taxi=np.stack(new_yellow, axis=0)
                uber_data=np.stack(new_uber, axis=0)
            elif cfg.granularity=='hour':
                all_data=all_data[90*24:(90+184)*24, :, :]
                green_taxi=green_taxi[90*24:(90+184)*24, :, :]
                yellow_taxi=yellow_taxi[90*24:(90+184)*24, :, :]
            # normalization
            green_taxi/=np.max(green_taxi)
            yellow_taxi/=np.max(yellow_taxi)
            uber_data/=np.max(uber_data)

        # split data
        T=all_data.shape[0]
        T1, T2 = int(T*0.8), int(T*0.9)
        train_data, dev_data, test_data=all_data[:T1,:,:], all_data[T1:T2,:,:], all_data[T2:T,:,:] # S_all, H, W
        if cfg.add_traffic:
            train_green, dev_green, test_green=green_taxi[:T1,:,:], green_taxi[T1:T2,:,:], green_taxi[T2:T,:,:]
            train_yellow, dev_yellow, test_yellow=yellow_taxi[:T1,:,:], yellow_taxi[T1:T2,:,:], yellow_taxi[T2:T,:,:]
            if cfg.year=='2014':
                train_uber, dev_uber, test_uber=uber_data[:T1,:,:], uber_data[T1:T2,:,:], uber_data[T2:T,:,:]
                self.train_data=np.stack((train_data, train_green, train_yellow, train_uber), axis=1) # S_train, 4, H, W
                self.dev_data=np.stack((dev_data, dev_green, dev_yellow, dev_uber), axis=1)
                self.test_data=np.stack((test_data, test_green, test_yellow, test_uber), axis=1)
            else:
                self.train_data=np.stack((train_data, train_green, train_yellow), axis=1) # S_train, 3, H, W
                self.dev_data=np.stack((dev_data, dev_green, dev_yellow), axis=1)
                self.test_data=np.stack((test_data, test_green, test_yellow), axis=1)
        else:
            self.train_data=np.expand_dims(train_data, axis=1)
            self.dev_data=np.expand_dims(dev_data, axis=1)
            self.test_data=np.expand_dims(test_data, axis=1)

        print('Reading data finished, training size:{}, dev size:{}, test size:{}'.\
            format(self.train_data.shape, self.dev_data.shape, self.test_data.shape))

    
    def reset(self):
        self.pointer=0
        self.end=False

    def sample(self,data='train', batch_size=4, seq_len=10, stride=1):
        if data=='train':
            train_size=self.train_data.shape[0]
            start_ids=np.random.choice(train_size-seq_len, size=batch_size)
            batch=[]
            for idx in start_ids:
                batch.append(self.train_data[idx:idx+seq_len, :,:,:])
            batch=np.stack(batch, axis=1) # S_batch, B, 3, H, W
            return batch
        else:
            test_data=self.dev_data if data=='dev' else self.test_data
            test_size=test_data.shape[0]
            batch=[]
            for b in range(batch_size):
                if self.pointer+seq_len>test_size:
                    self.end=True
                    break
                batch.append(test_data[self.pointer:self.pointer+seq_len,:,:,:])
                self.pointer+=stride
            batch=np.stack(batch, axis=1) # S, B, 3, H, W
            return batch
            
            
