import os
import torch
import torchaudio
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import get_files, check_dir
from feature_extraction import get_func_info, make_fixed_length

    
class AudioDataset(Dataset):
    def __init__(self, sample_dir, ext='.wav', fixed_sample_len=0):
        self.fixed_sample_len = fixed_sample_len # if 0 -> not fix the sample length for variable sample length input.

        self.sample_list = get_files(sample_dir, ext, True)
        self.labels = None
    
    
    def set_features(self, feature_name_list, kwargs_list):
        if type(feature_name_list) == str:
            feature_name_list = [feature_name_list]
        if type(kwargs_list) == dict:
            kwargs_list = [kwargs_list]
        
        assert (len(feature_name_list) == len(kwargs_list)) or len(kwargs_list) == 1
        # [kwargs <-> feature: one-to-one mapping] or [mixed_kwargs: share the arguments (arguments to each function are auto-selected)]
        function_list = [] # feature extraction function list
        target_kwargs_list = [] # kwargs list for each feature extraction function
        C_cumulation = [0] # feature channel(dim) cumulation
        seq_len = None # assume each feature sequence from a sample has same length (ignore fixed_seq_len at 1sample feature extraction; inference for variable length sample)

        # 1. get feature extraction function (from feature_extraction.py)
        for feature_idx, feature_name in enumerate(feature_name_list):
            feature_ext_function, target_kwargs, C, seq_len = get_func_info(feature_name, torch.randn((32000 if self.fixed_sample_len == 0 else self.fixed_sample_len),), kwargs_list[feature_idx % len(kwargs_list)])
            
            function_list.append(feature_ext_function) # feature extraction function list
            target_kwargs_list.append(target_kwargs) # kwargs corresponding each feature extraction function are extracted
            C_cumulation.append(C_cumulation[-1]+C) # feature dimension cumulation sum list (for list slicing)
            print(f'feature {feature_name_list[feature_idx]}: kwargs: {target_kwargs_list[feature_idx]}, dim: {C}')
        
        self.function_list = function_list
        self.kwargs_list = target_kwargs_list
        self.C_cumulation = C_cumulation
        if self.fixed_sample_len != 0:
            self.fixed_seq_len = seq_len #feature sequence length


    def convert_examples_to_features(self, wav_batch: torch.Tensor):
        # Extract features from waveform tensor. Each feature extracting methods are implemented in 'feature_extractor.py'.
        # 1. Batch tensor를 미리 메모리에 할당하기 위해, shape 계산.
        B, l = wav_batch.shape[0], wav_batch.shape[-1] # l: time-domain sequence length
        if hasattr(self, 'fixed_seq_len'):
            L = self.fixed_seq_len
        else:   
            L = 1 + (l - self.kwargs_list[0]['frame_len'])//self.kwargs_list[0]['shift_len']  # L: feature-domain sequence length.
        # feature tensor sequnce dim (B, C, L): L = 1 + (L - frame_len) // shift_len (when do not pad.)

        feature_tensor = torch.zeros((B, self.C_cumulation[-1], L), dtype=torch.float32) # one memory-allocation operation.
        for feature_idx, (feature_func, feature_kwargs) in enumerate(zip(self.function_list, self.kwargs_list)):
            feature_tensor[:, self.C_cumulation[feature_idx]: self.C_cumulation[feature_idx+1]] = feature_func(wav_batch, **feature_kwargs)
        
        return feature_tensor


    def real_time_extraction(self, sample_idx):
        sample = torchaudio.load(self.sample_list[sample_idx], normalize=True, channels_first=True)[0]
        if self.fixed_sample_len != 0:
            sample = make_fixed_length(sample, self.fixed_sample_len)
        if hasattr(self, 'function_list'):
            sample = self.convert_examples_to_features(sample)
        return sample


    def label_extraction(self):
        self.labels = torch.zeros((len(self.sample_list),), dtype=torch.long)
        for sample_idx, sample_file in enumerate(self.sample_list):
            spk_label = int(sample_file.split(os.sep)[-2])
            self.labels[sample_idx] = spk_label
        self.labels = torch.unique(self.labels, sorted=True, return_inverse=True)[1]
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, sample_idx):
        if self.labels != None:
            return self.real_time_extraction(sample_idx), self.labels[sample_idx]
        else:
            return self.real_time_extraction(sample_idx)


if __name__ == '__main__':
    train_dir = os.path.abspath('D:/SV/dataset/check')

    kwargs = {'sample_rate':16000, 'n_fft':400, 'frame_len':400, 'shift_len':160, 'n_mels':80, 'log_mel': False} # 25ms window, 10ms shift
    wav = get_files(train_dir, '.wav')[0]
    wav = torchaudio.load(wav, normalize=True, channels_first=True)[0]   
    print(wav.shape)
    
    # dataset = AudioDataset(train_dir, '.wav', 32000)
    # # dataset.label_extraction()
    # dataset.set_features('mel_spectrogram', kwargs)
    # dataloader = DataLoader(dataset, 8, True)
    
    # for X in dataloader:
    #     print(X.shape)
    # dataloader = DataLoader(dataset, 32, True)
    
    # for batch_idx, (X, Y) in enumerate(dataloader):
    #     if batch_idx == 2:
    #         break
    #     print(f'X: {X.shape}')
    #     print(f'Y: {Y.shape}')