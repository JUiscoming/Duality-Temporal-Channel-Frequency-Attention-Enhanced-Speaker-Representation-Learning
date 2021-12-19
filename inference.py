import yaml
import os
import pickle
#custom
from trainer import Trainer
from utils import get_files

best_epoch = 0

# 1. argument setting
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
    
# 1-1. argument data type
config['init_lr'] = float(config['init_lr'])
config['end_lr'] = float(config['end_lr'])
config['weight_decay'] = float(config['weight_decay'])

# 2. init trainer
tr = Trainer(config)

# 3. test
enroll_query_triplet = None
# 리스트이고 각 원소는 dev_dir의 wav file 경로를 뽑아냈을 때 file의 index를 기준으로 (enroll_idx, query_idx, True/False)

eer = tr.verify(best_epoch, enroll_query_triplet, 'test')
print(eer)


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