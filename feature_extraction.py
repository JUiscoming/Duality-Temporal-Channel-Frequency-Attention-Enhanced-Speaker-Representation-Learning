import torch
import torchaudio
import torchaudio.transforms as T
from inspect import signature
import random


def get_func_info(func_name, sample, kwargs):
    """[summary]

    Args:
        func_name (str): feature extraction function name
        sample (torch.tensor): a sample which is included in a dataset.
        kwargs (dict): keyword arguments dict which is covered the feature extraction function arguments.

    Returns:
        feature_dimension, feature_extraction_function_keyword_arguments
    """
    # 1. search the function named func_name in data_loader.py
    target_func = globals()[func_name]

    # 2. get target_func parameters except wav and get intersection args_name and kwargs.
    target_args_name = list(signature(target_func).parameters.keys())[1:] # param[0] = wav
    target_kwargs = {name: kwargs.get(name) for name in target_args_name}

    C, L = target_func(sample, **target_kwargs).shape[-2:]
    return target_func, target_kwargs, C, L


def mel_spectrogram(wav=None, sample_rate=16000, n_fft=640, frame_len=640, shift_len=160, n_mels=40, log_mel=False):
    # Log mel-filterbank energy (==log mel-spectrogram)
    # input: raw waveform(torch.Tensor), dim: (1, L)
    if not frame_len:
        frame_len = n_fft
    # output.shape: (B, n_mels, L)
    output = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=frame_len, hop_length=shift_len, n_mels=n_mels, center=False, window_fn=torch.hamming_window)(wav)
    if log_mel:
        output = torch.log(output + torch.finfo(torch.float32).eps)
    return output


def make_fixed_length(wav: torch.Tensor, target_length: int):
    if wav.dim() == 1:
        wav.unsqueeze(0) # batch-axis    
    # target_length보다 길이가 짧다면 cycle repeat로 길이를 맞춰줌. 더 긴 경우는 wav에서 random으로 target_length를 자름.
    if len(wav) < target_length:
        wav = cycle_repeat(wav, target_length)
    elif len(wav) > target_length:
        st = random.randint(0, len(wav)-target_length-1)
        wav = wav[st: st+target_length]
    
    return wav # wav.shape: (1, target_length)
    
    
def cycle_repeat(wav: torch.Tensor, target_length: int):
    if wav.dim() == 1:
        wav.unsqueeze(0) # batch-axis

    L = wav.shape[-1]
    multiple = target_length//L
    
    return wav.repeat(1, (multiple+1))[:, :target_length] # shape:(1, target_length)    
    
    
if __name__ == '__main__':
    # k = {'sample_rate':16000, 'n_fft':400, 'frame_len':400, 'shift_len':160, 'n_mels':80, 'log_mel': False}
    
    # sample = torch.randn((5, 32000)) # 1s
    # print(mel_spectrogram(sample, **k).shape)
    # L = (32000 - 400) // 160 + 1
    # print(f'est sample.L: {L}')
    
    # x = torch.randn((114688,))
    # L = (x.shape[-1] - 400) // 160 + 1
    # print(f'est x.L: {L}')
    # print(mel_spectrogram(x, **k).shape)
    
    x = torch.randn((32000,))
    s = cycle_repeat(x, 16000).shape
    s1 = make_fixed_length(x, 16000).shape
    
    print(s, s1)