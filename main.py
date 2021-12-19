import yaml
import os
import pickle
#custom
from trainer import Trainer

# 1. argument setting
with open('./config.yaml') as f:
    config = yaml.safe_load(f)
# 1-1. argument data type
config['init_lr'] = float(config['init_lr'])
config['end_lr'] = float(config['end_lr'])
config['weight_decay'] = float(config['weight_decay'])

# 2. init trainer
tr = Trainer(config)
# 3. train
tr.train()
# 4. test
with open(config['e_q_l_triplet_file'], 'rb') as f:
    enroll_query_label_triplet = pickle.load(f)
tr.verify(0, enroll_query_label_triplet, 'valid')