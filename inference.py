import yaml
import os
import pickle
#custom
from trainer import Trainer
from utils import get_files, enroll_query_label_triplet_extraction

best_epoch = 20
mode = 'test'

# 1. argument setting
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
    
# 1-1. argument data type
config['init_lr'] = float(config['init_lr'])
config['end_lr'] = float(config['end_lr'])
config['weight_decay'] = float(config['weight_decay'])

# 2. init trainer
tr = Trainer(config)

# 3. test: config['test_dir']에 있는 wav 파일 리스트를 추출한 후, trials.txt의 각 텍스트줄마다 utterance를 인덱스로 변환; (등록 발화 인덱스, 검사 발화 인덱스, label)
enroll_query_label_triplet = enroll_query_label_triplet_extraction(config[f'{mode}_dir'], '/home/disk2/Wook/data/Korean/test_trials.txt')
# 리스트이고 각 원소는 dev_dir의 wav file 경로를 뽑아냈을 때 file의 index를 기준으로 [(enroll_idx, query_idx, True/False)]
dict_output  = tr.verify(best_epoch, enroll_query_label_triplet, mode)
with open(f'{mode}_output.pkl', 'wb') as f:
    pickle.dump(dict_output, f, 4)