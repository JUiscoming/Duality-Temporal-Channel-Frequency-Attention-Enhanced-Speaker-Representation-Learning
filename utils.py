#-*- coding: utf-8 -*- 
# utils.py: custom functions and classes (logger and other functions)
import os
import datetime
import logging
import numpy as np
import random
import argparse
import torch
import inspect
import pickle
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from copy import deepcopy
from sklearn.metrics import roc_curve

##########################################
#           Initialize Logger            #
##########################################
def init_logger(log_dir=None, handler='both'):
    """Initialize logger.

    logging.getLogger(name): Multiple calls to getLogger() with the same name will always return a reference to the same Logger object. (in Python docs)
    So if you call 'logging.getLogger(same_name)' in other codes, it will return a reference to the same Logger object.
    
    To log message, use these methods; logger.info(msg) / logger.warning(msg) / logger.error(msg)

    Args:
        log_dir: if handler is set 'file' or 'both' the logs will be saved at log_dir. Also it is used to identify unique logger (str, optional).
        handler: print the logs at designated places. file: txt / stream: console / both: file and stream. defaults to 'both' (str, optional).

    Returns:
        logger instance
    """
    assert handler in ['file', 'stream', 'both']
    if not log_dir:
        log_dir = os.path.abspath('')
    check_dir(log_dir)

    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO) # message below the setLevel will be ignored.
    # Formatter; the format of the log(print)
    formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')

    if (handler == 'file') or (handler == 'both'):
        init_time = datetime.datetime.now().strftime('%Y.%m.%d %H-%M-%S')
        fname = f'log_{init_time}.log'
        file = os.path.join(log_dir, fname)

        file_handler = logging.FileHandler(filename = file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if (handler == 'stream') or (handler == 'both'):
        # Stream(Console) Handler; Handler object dispatchs specific message for proper levels to specific point like a console or a file.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


##########################################
#        Process files and dirs          #
##########################################
def get_files(paths, ext = None, search_subdir = True):
    # paths: target paths (str or list) 
    # ext: get files which has specified extension. (str)
    # search_subdir: whether find files which exist sub-directory. (bool)
    if ext and ext[0] != '.':
        ext = '.' + ext

    target_files = []
    if type(paths) == str:
        paths = [paths]
    
    # 1. get files
    for path in paths:
        if not ext:
            if search_subdir:
                for root, dirs, files in os.walk(path):
                    target_files += [os.path.join(root, file) for file in files]
            else:
                target_files += [os.path.join(path, file) for file in os.listdir(path)]
        else:
            if search_subdir:
                for root, dirs, files in os.walk(path):
                    target_files += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1] == f'{ext}']
            else:
                target_files += [os.path.join(path, file) for file in os.listdir(path) if os.path.splitext(file)[-1] == f'{ext}']
    
    # 2. exclude ._ file (in MAC)
    target_files = [path for path in target_files if os.path.split(path)[-1][:2] != "._"]

    # 3. sort by filenames
    target_files.sort(key = lambda x: os.path.split(x)[-1])
    
    return target_files


def check_dir(paths):
    # Check whether the directories exist. If doesn't exist, make the directories.
    if (type(paths) != list) and (type(paths) != tuple):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def get_filename(path: str):
    # get filename without extension.
    return os.path.splitext(os.path.split(path)[-1])[0]


##########################################
#           Process strings              #
##########################################
def separate_str_comma(s):
    # Separate a string based on comma.
    return s.replace(" ","").split(',')


def find_word_all(source, target):
    # find all words (target) from source and return the start index of each word.
    index = -1
    result = []

    while True:
        index = source.find(target, index+1)
        
        if index == -1:
            break
        result.append(index)
    
    return result


def str2bool(v): 
    # map from str to bool
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    """Map from a string to list. If brackets are exists in the string, separate to sublist.

    Args:
        s: the target strings to map list (str, required).

    Examples:
        >>> s = '0,1,2,3,4,5'
        >>> out = str2list(s)
        >>> out == ['0', '1', '2', '3', '4', '5']

    Alternatively, when brackets are exists in the string:
        >>> s = '(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1)'
        >>> out = str2list(s)
        >>> out == [['0','1','2','3','4'], ['1','2','3','4','0'], ['2','3','4','0','1']]
    """
    if s == 'none' or s == '':
        return False

    s = s.replace(" ", "")
    left_brackets = find_word_all(s, '(')
    right_brackets = find_word_all(s, ')')
    if (not left_brackets) or (not right_brackets):
        s = s.replace(")", "")
        s = s.replace("(", "")
        return separate_str_comma(s)

    result = []
    for left, right in zip(left_brackets, right_brackets):
        result.append(separate_str_comma(s[left+1: right]))
    return result


##########################################
#            Plot functions              #
##########################################
def plot_confusion_matrix(cm, file=None, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    f = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if not file:
        plt.show()
    else:
        plt.savefig(file, dpi=160)


##########################################
#           Other functions              #
##########################################
def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def dict_str_pretty(d):
    result = ""
    for k, v in d.items():
        result += f"{k}: {v}\n"
    return result


def args_str_pretty(args):
    result = ""
    for k in vars(args):
        result += f"{k}: {getattr(args, k)}\n"
    return result


class EarlyStopper():
    """Stop training when a monitored metric has stopped improving.
    min_delta (float): Minimum change in the monitored quantity to qualify as an improvement,
    i.e. an absolute change of less than min_delta, will count as no improvement.
    patience (int): Number of epochs with no improvement after which training will be stopped.
    mode (str): One of {"min", "max"}.
    In min mode, training will stop when the quantity monitored has stopped decreasing;
    in "max" mode it will stop when the quantity monitored has stopped increasing.
    """
    def __init__(self, min_delta=0, patience=0, mode='min'):
        assert mode in ['min', 'max']
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode

        self.best = None
        self.count = 0
        
    def __call__(self, criteria: float):
        # criteria (float): Quantity to be monitored (required).
        # self.count 
        if self.best == None:
            self.best = criteria
        else:
            condition = (criteria > (self.best + self.min_delta)) if self.mode == 'max' else (criteria < (self.best - self.min_delta))
            if condition:
                self.best = criteria
                self.count = 0
            else:
                self.count += 1
        
        return self.count


def compute_eer(distances, labels):
    fprs, tprs, _ = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return eer
 

class ValueSaver:
    def __init__(self):
        self.memory = dict()

    def add_item(self, epoch_index, key, value):
        # 동일한 epoch_index 얻어지는 key-value pair 저장 (bind_index 단위는 보통 epoch)
        # 한 epoch에서 얻는 loss나 metric들을 하나의 리스트에 보관. 이후 mean과 같은 통계량 계산 가능.
        if epoch_index not in self.memory: # 처음 epoch_index 저장하는 경우.
            self.memory[epoch_index] = defaultdict(list)
            
        if type(value) == list: # value가 리스트인 경우, concat
            self.memory[epoch_index][key] += value
        else: # value가 scalar인 경우, append
            self.memory[epoch_index][key].append(value)

    def add_dict(self, epoch_index, dict_sample):
        for key, value in dict_sample.items():
            self.add_item(epoch_index, key, value)

    def get_mean(self, epoch_index, key):
        return np.mean(self.memory[epoch_index][key])
    
    def get_std(self, epoch_index, key):
        return np.std(self.memory[epoch_index][key])

    def get_max(self, epoch_index, key):
        return max(self.memory[epoch_index][key])

    def get_min(self, epoch_index, key):
        return min(self.memory[epoch_index][key])

    def get_item(self, epoch_index, key):
        return self.memory[epoch_index][key]

    def __getitem__(self, epoch_index):
        return self.memory[epoch_index]


def enroll_query_label_triplet_extraction(dev_dir, txt_file, output_file):
    wav_file_list = get_files(dev_dir, '.wav', True)
    wav_file_dict = {os.path.splitext(os.path.basename(wav_file))[0][6:]: idx for idx, wav_file in enumerate(wav_file_list)}
    triplet_list = []
    not_exist_file_txt = 'D:\\SV\\dataset\\not_exist.txt'
    not_ex_f = open(not_exist_file_txt, 'w')
    line_idx = 0

    with open(txt_file, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            
            enroll, query, label = line.rstrip().split(' ')
            label = True if label[0] == 't' else False
            try: enroll = wav_file_dict[enroll]
            except: not_ex_f.write(f'{line_idx}: {enroll}\n')
            else:
                try: query = wav_file_dict[query]
                except: not_ex_f.write(f'{line_idx}: {query}\n')
                else: triplet_list.append((enroll, query, label))
            line_idx += 1
    
    with open(output_file, 'wb') as f:
        pickle.dump(triplet_list, f, 4)
    not_ex_f.close()


if __name__ == '__main__':
    dev_dir = 'D:\\SV\\dataset\\dev'
    txt_file = 'D:\\SV\\dataset\\trials.txt'
    enroll_query_label_triplet_extraction(dev_dir, txt_file, 'D:\\SV\\dataset\\valid.pkl')