# Duality-Temporal-Channel-Frequency-Attention-Enhanced-Speaker-Representation-Learning
Unofficial implementation of Duality Temporal Channel Frequency Attention Enhanced Speaker Representation Learning arXiv https://arxiv.org/abs/2110.06565

I used this model at hackathon (speaker verification). If you want to use this code, you need to modify some part of code.

- label_extraction at data_loader.py
- enroll_query_label_triplet_extraction at utils.py

## Training

### 1. Set the config.yaml

- 'train_dir': waveform files in the 'train_dir' are loaded.

- 'valid_dir', 'test_dir': It's the same as 'train_dir'.

- If you want to use other features, make the feature extract function at feature_extraction.py and then set 'feature_name_list': [feature_func_name1, feature_func_name2, ...] (all features are assumed same sequence length.)

- If you use more than 2 features, you declair the 'feature_kwargs' to list. For each feature extraction function, arguments can be provided in the form of a list of dictionary or by putting all parameters in one dictionary.

### 2. modify main.py

- Until line 18 [tr.train()], the main.py will work without any problems.

### 3. extract enroll_query_label_triplet for Trainer().verify()
- If you want to verify for validation set or test set, some information is needed.

- Trainer().verify(epoch, enroll_query_label_triplet, mode)

- epoch: load the model trained for config['epoch']

- mode: 'valid' -> config['valid_dir'], 'test' -> config['test_dir']

- enroll_query_label_triplet: here, the index refers to the index of the list in which wav in the config[f'{mode}_dir'] is sorted by the filename.
enroll_query_label_triplet is a list that the elements are a tuple (enrollment_utterance_index, query_utterance_index, True/False(correspondence)) or (enrollment_utterance_index, query_utterance_index).
In the former case, information such as equal error rate (EER) is returned, and in the latter case, only the similarity score is returned.