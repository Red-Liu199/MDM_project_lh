import torch
import numpy as np
from config import global_config as cfg
from model import ConvLSTM_net, STTransformer
from reader import traffic_reader
import shutil, os
from tqdm import tqdm
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
class run():
    def __init__(self, mode='train', path=None):
        if mode=='train':
            if cfg.model=='convlstm':
                self.model=ConvLSTM_net()
            elif cfg.model=='transformer':
                self.model=STTransformer(cfg.input_channels,embed_size=cfg.embed_size, time_num=50, num_layers=cfg.layers, 
                    T_dim=cfg.in_len, output_T_dim=cfg.out_len,heads=cfg.heads, forward_expansion=2)
            self.optimizer=torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
            self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_step_size, gamma=cfg.gamma)
            
            self.exp_path=os.path.join('experiments', cfg.exp_name)
            log_path=os.path.join(self.exp_path, 'train_log')
            if not os.path.exists(self.exp_path):
                os.mkdir(self.exp_path)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        elif mode=='test':
            self.exp_path=path
            config=json.load(open(os.path.join(path, 'config.json'), 'r'))
            for key, value in config.items():
                cfg.__setattr__(key, value)
            if cfg.model=='convlstm':
                self.model=ConvLSTM_net()
            elif cfg.model=='transformer':
                self.model=STTransformer(cfg.input_channels,embed_size=cfg.embed_size, time_num=50, num_layers=cfg.layers, 
                    T_dim=cfg.in_len, output_T_dim=cfg.out_len,heads=cfg.heads, forward_expansion=2)
            self.model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth')))
        
        self.model.to(cfg.device)
        self.reader=traffic_reader()
        
    def calculate_loss(self, output, target, return_loss=True):
        # output: S*B*C*H*W, but we only calculate the first channel
        # error: S*B
        mse = torch.sum((output[:,:,0,:,:]-target[:,:,0,:,:])**2, (2, 3))
        mae = torch.sum(torch.abs((output[:,:,0,:,:]-target[:,:,0,:,:])), (2, 3))
        if return_loss:
            return torch.mean(mse) + torch.mean(mae)
        else:
            return torch.sum(mse, (0, 1)), torch.sum(mae, (0, 1))

    def train(self):
        print('*********Start training*********')
        json.dump(cfg.__dict__, open(os.path.join(self.exp_path,'config.json'),'w'),indent=2)
        train_loss = 0.0
        min_loss=100000
        valid_loss=self.validate(data='dev')
        print('Initial validation loss:{}'.format(valid_loss))
        for itera in tqdm(range(1, cfg.max_iterations+1)):
            train_batch=self.reader.sample(data='train', batch_size=cfg.batch_size, seq_len=cfg.in_len+cfg.out_len)
            train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.device)
            if itera==1:
                print('batch size:', train_batch.shape)
            train_data = train_batch[:cfg.in_len, ...]
            train_label = train_batch[cfg.in_len:cfg.in_len + cfg.out_len, ...]
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.calculate_loss(output, train_label)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=50.0)
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()
            self.tb_writer.add_scalar('step_train_loss', loss.item(), itera)

            if itera % cfg.test_iteration_interval == 0:
                valid_loss=self.validate(data='dev')
                print('Iteration:{}, validation loss:{}, training loss:{}'.format(itera, valid_loss, train_loss))
                self.tb_writer.add_scalar('valid_loss', valid_loss, itera)
                self.tb_writer.add_scalar('epoch_train_loss', train_loss, itera)
                train_loss=0
                if valid_loss<min_loss:
                    min_loss=valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.exp_path, 'best_model.pth'))
                    print('Model saved in {}'.format(self.exp_path))

                    
    def validate(self, data='dev'):
        self.reader.reset()
        self.model.eval()
        valid_loss=0
        with torch.no_grad():
            while not self.reader.end:
                valid_batch=self.reader.sample(data=data, batch_size=cfg.batch_size, seq_len=cfg.in_len+cfg.out_len)
                if valid_batch.shape[1] == 0:
                    break
                valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.device)
                valid_data = valid_batch[:cfg.in_len, ...]
                valid_label = valid_batch[cfg.in_len:cfg.in_len + cfg.out_len, ...]
                output = self.model(valid_data)
                loss = self.calculate_loss(output, valid_label)
                valid_loss += loss.item()
        return valid_loss
    
    def test(self, data='test'):
        self.reader.reset()
        self.model.eval()
        MSE, MAE = 0, 0
        count=0
        pred_samples, true_samples=[], []
        cfg.out_len=1
        with torch.no_grad():
            while not self.reader.end:
                valid_batch=self.reader.sample(data=data, batch_size=cfg.batch_size, seq_len=cfg.in_len+cfg.out_len)
                if valid_batch.shape[1] == 0:
                    break
                valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.device)
                valid_data = valid_batch[:cfg.in_len, ...]
                valid_label = valid_batch[cfg.in_len:cfg.in_len + cfg.out_len, ...]
                output = self.model(valid_data)
                while len(pred_samples)<24:
                    pred_samples.append(output[:,:,0,:,:].reshape(-1, output.shape[-2], output.shape[-1]).cpu().numpy())
                    true_samples.append(valid_label[:,:,0,:,:].reshape(-1, output.shape[-2], output.shape[-1]).cpu().numpy())
                mse, mae = self.calculate_loss(output, valid_label, return_loss=False)
                MSE+=mse
                MAE+=mae
                count+=output.shape[0]*output.shape[1] # seq_len*batch_size
        print('Total prediction periods:{}'.format(count))
        print('Area size for each period:{}'.format((output.shape[-2], output.shape[-1])))
        print('Average MSE:{}, average MAE:{}'.format(MSE/count, MAE/count))
        # plot figures
        pred_samples=np.concatenate(pred_samples, axis=0)
        true_samples=np.concatenate(true_samples, axis=0)
        plt.figure(figsize=(6, 12))
        for i in range(4):
            for j in range(2):
                idx=i*2+j
                plt.subplot(4,2,idx+1)
                plt.imshow(pred_samples[idx,:,:])
        plt.savefig(os.path.join(self.exp_path, 'pred_heatmap.png'))
        plt.figure(figsize=(6, 12))
        for i in range(4):
            for j in range(2):
                idx=i*2+j
                plt.subplot(4,2,idx+1)
                plt.imshow(true_samples[idx,:,:])
        plt.savefig(os.path.join(self.exp_path, 'true_heatmap.png'))
        plt.figure()
        plt.imshow(pred_samples[0,:,:])
        plt.colorbar()
        plt.savefig(os.path.join(self.exp_path, 'pred_heatmap_single.png'))
        plt.figure()
        plt.imshow(true_samples[0,:,:])
        plt.colorbar()
        plt.savefig(os.path.join(self.exp_path, 'true_heatmap_single.png'))
        print('Ploting samples finished')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train')
    parser.add_argument('-path', default=None)
    args = parser.parse_args()
    if args.mode=='train':
        R=run()
        R.train()
    elif args.mode=='test':
        R=run(mode='test', path=args.path)
        R.test()
    
