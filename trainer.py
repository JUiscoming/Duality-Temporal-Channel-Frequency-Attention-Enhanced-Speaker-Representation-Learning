# non data-science
from tqdm import tqdm
import pickle
import os
# data-science
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score # cm, cr, WA, UA, F1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# custom
from utils import init_logger, check_dir, fix_seed, dict_str_pretty, EarlyStopper, ValueSaver, compute_eer
from data_loader import AudioDataset
from loss import LossFunction
from models import ResNet_DTCF


class Trainer(object):
    def __init__(self, config):
        # 1. set the directories (all experiments directories are nested in exp_dir.)
        os.chdir(os.path.dirname(os.path.abspath(__file__))) # change working directory
        
        config['exp_dir'] = os.path.abspath(os.path.join('exps', config['exp_id'])) # All files generated in a experiment are located in this directory
        config['log_dir'] = os.path.join(config['exp_dir'], 'logs') # log dir
        config['tboard_dir'] = os.path.join(config['exp_dir'], 'tensorboard') # Tensorboard log dir
        config['model_dir'] = os.path.join(config['exp_dir'], 'models') # Model dir
        config['output_dir'] = os.path.join(config['exp_dir'], 'outputs') # Output (etc: confusion matrices, samples, ...) dir
        check_dir((config['exp_dir'], config['log_dir'], config['tboard_dir'], config['model_dir'], config['output_dir']))
        self.config = config

        # 2. initialize random seed value.
        fix_seed(config['seed'])

        # 3. Initialize logger.
        self.logger = init_logger(self.config['log_dir'], handler='file') # Handler: 'file' (save log at args.log_dir), 'stream' (print log at console), 'both'
        self.logger.info('***** Experiment.config *****') # Save the experiment arguments
        self.logger.info(dict_str_pretty(self.config))

        # 4. Modules
        self.dataset = dict()
        self.dataloader = dict()

        # 5. Tensorboard
        self.writer = SummaryWriter(config['tboard_dir'])
        self.writer.add_text('config', str(self.config)) # save the used arguments at tensorboard

        # 6. Set devices
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
        os.environ["CUDA_VISIBLE_DEVICES"]= self.config['gpu_idx']  # Set the GPUs
        self.device = "cuda" if torch.cuda.is_available() and self.config['gpu_idx'] else "cpu"
        if self.device == "cuda":
            self.logger.info(f'***** CUDA enviroments *****')
            for device_idx in range(torch.cuda.device_count()):
                self.logger.info('Index {:>2}: total memory = {:.3f} GB ; name = {}'.format(device_idx,
                torch.cuda.get_device_properties(device_idx).total_memory/(1024**3),torch.cuda.get_device_properties(device_idx).name))
        self.logger.info(f'Device type: {self.device}' + (f", selected GPU indices: {self.config['gpu_idx']}" if self.device == 'cuda' else ''))

        # 7. Loss function
        self.AAM_Softmax = LossFunction(self.config['emb_dim'], self.config['n_spks'], self.config['margin'], self.config['scale']).to(self.device)
        
        # 8. Init model
        self.init_model()
        self.model_to_gpu()
       
        
    def __del__(self):
        self.writer.close()


    def multi_gpu(self, gpu_idx):
        gpu_idx = [int(idx) for idx in gpu_idx.split(',')]
        for key in self.models:
            self.models[key] = nn.DataParallel(self.models[key], device_ids=gpu_idx)        

    ##### train/verify step     #####
    ##### 1) training step      #####
    ##### 2) verifying step     #####
    #################################
    def training_step(self, batch, batch_idx): # batch_idx에 따라서 다른 작업을 수행할 수 있음.
        for key in self.models.keys():
            self.models[key].train()

        # 1. input and label
        X, Y = batch
        X, Y = X.to(self.device), Y.to(self.device)

        # 2. calculate output and losses
        embedding = self.models['SV'](X)
        loss_aam_softmax, accuracy = self.AAM_Softmax(embedding, Y)
        if not torch.isfinite(loss_aam_softmax): # Are the loss value finite?
            self.logger.warning(f'[train] Restart training from the epoch 0 due to nan loss at loss_aam_softmax.')
            return None
        
        self.optimizer['SV'].zero_grad() # 이전 iteration에서 계산된 gradient의 영향을 받지 않기 위해 zero_grad 선행.
        loss_aam_softmax.backward()
        nn.utils.clip_grad_norm_(self.models['SV'].parameters(), self.config['max_grad_norm'])
        self.optimizer['SV'].step()
 
        # 3. calculate metrics and detach all return values
        # actual = Y.detach().cpu().numpy().tolist()
        # predicted = torch.argmax(spk_logit, dim=-1).detach().cpu().numpy().tolist() # (B, 4)에서 마지막 축의 max값 index
        
        loss_aam_softmax = loss_aam_softmax.mean().item()
        accuracy = accuracy.item()

        return {'train/loss/aam_softmax': loss_aam_softmax, 'train/metric/accuracy': accuracy}
    
    
    def verification_step(self, enrollment, query):        
        for key in self.models.keys():
            self.models[key].eval()
        
        with torch.no_grad(): # gradient 계산 x
            # 1. enrollment utterance, query utterance
            enrollment, query = enrollment.to(self.device), query.to(self.device)

            # 2. calculate output and losses
            embedding_e = self.models['SV'](enrollment)
            embedding_q = self.models['SV'](query)
            
            # 3. calculate similarity btw. two embeddings
            # score = torch.cdist(embedding_e, embedding_q, p=2)
            score = 1-F.cosine_similarity(embedding_e, embedding_q)
            
        return score.item()


    ##### training/verification #####
    ##### 1) train              #####
    ##### 2) verify             #####
    #################################            
    def train(self):
        self.init_dataset('train')
        self.init_optim_and_lrs()
        if len(self.config['gpu_idx'].split(',')) > 1:
            self.multi_gpu(self.config['gpu_idx'])
        self.dataloader['train'] = DataLoader(self.dataset['train'], self.config['batch_size'], True, num_workers=self.config['num_workers'])
        # valid utterance pair information
        with open(self.config['e_q_l_triplet_file'], 'rb') as f:
            enroll_query_label_triplet = pickle.load(f)

        if self.config['loaded_epoch'] > 0:
            value_saver = self.load_models(self.config['loaded_epoch'])
        else:
            value_saver = ValueSaver() # save loss, metric
        log_step = len(self.dataloader['train']) if self.config['log_num'] == 0 else len(self.dataloader['train']) // self.config['log_num']

        
        start_epoch = (self.config['loaded_epoch'] + 1) if self.config['loaded_epoch'] > 0 else 1
        epoch_iterator = tqdm(range(start_epoch, self.config['epoch']+1), initial=start_epoch-1, total=self.config['epoch'], desc="Epoch", ascii=True, mininterval=2)
        # initial: 전체 작업중 이미 처리된 작업 수
        # total: 전체 작업 수
        for epoch in epoch_iterator:
            # 1. training session
            train_step_iterator = tqdm(self.dataloader['train'], desc="Iteration", ascii=True, mininterval=2) # update every 2 second
            for train_step_idx, (X, Y) in enumerate(train_step_iterator):
                # X: feature, Y: speaker_label
                output = self.training_step(batch=[X, Y], batch_idx=train_step_idx)
                if not output:
                    return -1 # return value == ended_epoch (if return value == -1, it means you need to re-training)
                value_saver.add_dict(epoch_index=epoch, dict_sample=output)
                epoch_iterator.set_postfix({'spk_loss': f"{value_saver.get_mean(epoch, 'train/loss/aam_softmax'):.6f}", 'acc': f"{value_saver.get_mean(epoch, 'train/metric/accuracy'):.6f}"})
                if (train_step_idx+1) % log_step == 0:
                    msg = f"[train E:{epoch}/S:{train_step_idx}] spk_loss: {value_saver.get_mean(epoch, 'train/loss/aam_softmax'):.6f}, accuracy: {value_saver.get_mean(epoch, 'train/metric/accuracy'):.6f}"
                    self.logger.info(msg)
            
            # 2. valid session
            if self.config['valid_step'] != 0 and epoch % self.config['valid_step'] == 0:
                dict_output = self.verify(epoch, enroll_query_label_triplet, 'valid') 
                value_saver.add_dict(epoch, dict_output)
        
            # 3. lrs step
            if epoch % self.config['step_size'] == 0:
                for lrs_key in self.lrs:
                    self.lrs[lrs_key].step()
            
            # 4. loss, metric, lr write at tensorboard
            # 4-1. train (lr, loss, WA, UA, F1, lr)
            for optim_key in self.optimizer:
                value_saver.add_item(epoch, f'train/lr/{optim_key}', self.optimizer[optim_key].param_groups[0]["lr"])
            
            # 4-2. upload data at tensorboard
            for k, v in value_saver[epoch].items():
                pass_key = ['actual', 'predicted', 'score', 'label']
                while pass_key:
                    if pass_key.pop() in k.split('/'):
                        break
                if pass_key:
                    continue
                self.writer.add_scalar(k, value_saver.get_mean(epoch, k), epoch)

            # 5. save model
            if epoch % self.config['save_step'] == 0:
                self.save_models(epoch)
                self.save_others(epoch, value_saver)               
                     
                
    def verify(self, epoch, enroll_query_label_triplet, mode='valid'):
        assert mode == 'valid' or mode == 'test'
        # enrollment_utterance & query_utterance pair indices at self.dataset[sample_idx]; [(e_index0, q_index0, True), (e_index1, q_index1, False), ...]
        
        if mode == 'valid' and self.config['loaded_epoch'] > 0:
            self.load_models(self.config['loaded_epoch'])
        if mode == 'test' and epoch > 0:
            self.load_models(epoch)
                
        self.init_dataset(mode)
        dataset = self.dataset[mode]
        
        score_list, label_list = [], []
        
        verify_step_iterator = tqdm(enroll_query_label_triplet, desc="Iteration", ascii=True, mininterval=2)
        
        # label is given
        if len(enroll_query_label_triplet[0]) == 3:
            for step_idx, (e_idx, q_idx, label) in enumerate(verify_step_iterator):
                score = self.verification_step(dataset[e_idx], dataset[q_idx])
                score_list.append(score)
                label_list.append(label)
            eer = compute_eer(score_list, label_list)
            
            return {f'{mode}/eer': eer}
        
        # label is not given
        for step_idx, (e_idx, q_idx) in enumerate(verify_step_iterator):
            score = self.verification_step(dataset[e_idx], dataset[q_idx])
            score_list.append(score)
        
        return {f'{mode}/score': score_list}
            
                  

    ##### save & load & remove  #####
    ##### 1) model              #####
    ##### 2) optim & lrs        #####
    #################################
    def save_models(self, epoch):
        state = {}
        for key, model in self.models.items():
            if len(self.config['gpu_idx'].split(',')) <= 1:
                state[key] = model.state_dict()
            else:
                state[key] = model.module.state_dict()
        filename = f'model_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        torch.save(state, file)
    

    def load_models(self, epoch):
        filename = f'model_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        if os.path.isfile(file):
            state = torch.load(file, map_location='cpu')
            for key in self.models.keys():
                self.models[key].load_state_dict(state[key])
            return True
        else:
            return False


    def remove_model(self, epoch):
        filename = f'model_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        os.remove(file)
    

    def save_others(self, epoch: int, value_saver=None):
        # others: optimizer_state_dict, lr_scheduler_state_dict
        state = {"epoch": epoch,
                 "optimizer": {},
                 "lrs": {},
                 "value_saver": value_saver}

        for key, opt in self.optimizer.items():
            state['optimizer'][key] = opt.state_dict()
        for key, lrs in self.lrs.items():
            state['lrs'][key] = lrs.state_dict()
        
        filename = f'others_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        torch.save(state, file)
    

    def load_others(self, epoch: int):
        filename = f'others_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        state = torch.load(file, map_location='cpu')

        for key in self.optimizer.keys():
            self.optimizer[key].load_state_dict(state['optimizer'][key])
        for key in self.lrs.keys():
            self.lrs[key].load_state_dict(state['lrs'][key])
        if state["value_saver"] != None:
            return state["value_saver"]


    def remove_others(self, epoch):
        filename = f'others_e{epoch}.pt'
        file = os.path.join(self.config['model_dir'], filename)
        os.remove(file)
        

    ##### Module initialization #####
    ##### 1) dataset            #####
    ##### 2) model              #####
    ##### 3) model_to_gpu       #####
    ##### 4) optim_and_lrs      #####
    #################################
    def init_dataset(self, mode='train'):
        assert mode in ['train', 'valid', 'test']
        self.dataset[mode] = AudioDataset(self.config[f'{mode}_dir'], '.wav', self.config['fixed_sample_len'] if mode == 'train' else 0)    
        self.dataset[mode].set_features(self.config['feature_name_list'], self.config['feature_kwargs'])       
        if mode == 'train':
            self.dataset[mode].label_extraction()
        
        
    def init_model(self):
        self.models = dict()
        self.models['SV'] = ResNet_DTCF(
            num_blocks=[3, 4, 6, 3], num_filters=[32, 32, 64, 128, 256],
            emb_dim=self.config['emb_dim'], n_mels=self.config['feature_kwargs']['n_mels'])
        
        
    def model_to_gpu(self):
        for key in self.models.keys():
            self.models[key].to(self.device)


    def init_optim_and_lrs(self):
        self.optimizer = dict()
        self.lrs = dict()
        params = list(self.models['SV'].parameters()) + list(self.AAM_Softmax.parameters())
        self.optimizer['SV'] = optim.Adam(params=params, lr=self.config['init_lr'], weight_decay=self.config['weight_decay'])
        gamma = (self.config['end_lr'] / self.config['init_lr']) ** (1/(self.config['epoch']-1))
        self.lrs['SV'] = optim.lr_scheduler.StepLR(self.optimizer['SV'], self.config['step_size'], gamma)
        
            
if __name__ == '__main__':
    distance = np.array([0.15, 0.17, 0.23, 0.25, 0.31, 0.43, 0.5])
    label = np.array([True, True, False, True, False, False, False])
    print(compute_eer(distance, label))