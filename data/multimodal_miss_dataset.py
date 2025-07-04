import os
import json
import random
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class MultimodalMissDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='IEMOCAP', help='which dataset to use')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.curriculum_stg = opt.curriculum_stg
        self.Is_part_miss = opt.Is_part_miss
        self.miss_rate = opt.miss_rate
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', f'{opt.corpus_name}_config.json')))
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        # load feature
        self.A_type = opt.A_type
        self.all_A = self.h5_to_dict(
            h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}.h5'), 'r'))
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        elif self.A_type == 'comparE_raw':
            self.mean, self.std = self.calc_mean_std()
            
        self.V_type = opt.V_type
        self.all_V = self.h5_to_dict(
            h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}.h5'), 'r'))
        self.L_type = opt.L_type
        self.all_L = self.h5_to_dict(
            h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}.h5'), 'r'))
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        if self.corpus_name == 'IEMOCAP':
            self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)
        # make missing index
        if set_name != 'trn':           # val && tst
            if 'CL' in opt.model:
                if self.curriculum_stg == 'single':
                    self.missing_index = torch.tensor([
                        [1, 1, 0],  # AVZ
                        [1, 0, 1],  # AZL
                        [0, 1, 1],  # ZVL
                    ] * len(self.label)).long()
                    self.miss_type = ['avz', 'azl', 'zvl'] * len(self.label)
                elif self.curriculum_stg == 'multiple':
                    self.missing_index = torch.tensor([
                        [1, 0, 0],  # AZZ
                        [0, 1, 0],  # ZVZ
                        [0, 0, 1],  # ZZL
                    ] * len(self.label)).long()
                    self.miss_type = ['azz', 'zvz', 'zzl'] * len(self.label)
                else:
                    self.missing_index = torch.tensor([
                        [1, 0, 0],  # AZZ
                        [0, 1, 0],  # ZVZ
                        [0, 0, 1],  # ZZL
                        [1, 1, 0],  # AVZ
                        [1, 0, 1],  # AZL
                        [0, 1, 1],  # ZVL
                    ] * len(self.label)).long()
                    self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'] * len(self.label)
            else:
                self.missing_index = torch.tensor([
                                                      [1, 0, 0],  # AZZ
                                                      [0, 1, 0],  # ZVZ
                                                      [0, 0, 1],  # ZZL
                                                      [1, 1, 0],  # AVZ
                                                      [1, 0, 1],  # AZL
                                                      [0, 1, 1],  # ZVL
                                                  ] * len(self.label)).long()
                self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'] * len(self.label)
        else:                           # trn
            if 'CL' in opt.model:
                if self.curriculum_stg == 'single':
                    self.missing_index = [
                        [1, 1, 0],  # AVZ
                        [1, 0, 1],  # AZL
                        [0, 1, 1],  # ZVL
                    ]
                    self.miss_type = ['avz', 'azl', 'zvl']
                elif self.curriculum_stg == 'multiple':
                    self.missing_index = [
                        [1, 0, 0],  # AZZ
                        [0, 1, 0],  # ZVZ
                        [0, 0, 1],  # ZZL
                    ]
                    self.miss_type = ['azz', 'zvz', 'zzl']
                else:
                    self.missing_index = [
                        [1, 0, 0],  # AZZ
                        [0, 1, 0],  # ZVZ
                        [0, 0, 1],  # ZZL
                        [1, 1, 0],  # AVZ
                        [1, 0, 1],  # AZL
                        [0, 1, 1],  # ZVL
                    ]
                    self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
            else:
                self.missing_index = [
                    [1, 0, 0],  # AZZ
                    [0, 1, 0],  # ZVZ
                    [0, 0, 1],  # ZZL
                    [1, 1, 0],  # AVZ
                    [1, 0, 1],  # AZL
                    [0, 1, 1],  # ZVL
                ]
                self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
        
        # set collate function
        self.manual_collate_fn = True
    
    def h5_to_dict(self, h5f):
        ret = {}
        for key in h5f.keys():
            ret[key] = h5f[key][()]
        return ret
    
    def __getitem__(self, index):
        if self.set_name != 'trn':
            if self.curriculum_stg in ['single', 'multiple']:
                feat_idx = index // 3
            else:
                feat_idx = index //6
            if self.Is_part_miss:
                if random.random() < self.miss_rate:
                    missing_index = self.missing_index[index]
                    miss_type = self.miss_type[index]
                else:
                    missing_index = torch.tensor([1, 1, 1])
                    miss_type = 'avl'
            else:
                missing_index = self.missing_index[index]
                miss_type = self.miss_type[index]
        else:
            feat_idx = index
            if self.Is_part_miss:
                if random.random() < self.miss_rate:
                    missing_index = torch.tensor(random.choice(self.missing_index)).long()
                    miss_type = random.choice(self.miss_type)
                else:
                    missing_index = torch.tensor([1, 1, 1])
                    miss_type = 'avl'
            else:
                missing_index = torch.tensor(random.choice(self.missing_index)).long()
                miss_type = random.choice(self.miss_type)
        
        int2name = self.int2name[feat_idx]
        if self.corpus_name == 'IEMOCAP':
            int2name = int2name[0].decode()
        label = torch.tensor(self.label[feat_idx])
        
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE' or self.A_type == 'comparE_raw':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        } if self.set_name == 'trn' else{
            'A_feat': A_feat * missing_index[0], 
            'V_feat': V_feat * missing_index[1],
            'L_feat': L_feat * missing_index[2],
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }
    
    def __len__(self):
        return len(self.missing_index) if self.set_name != 'trn' else len(self.label)

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float() 
        std_f = torch.std(features, dim=0).unsqueeze(0).float()  
        std_f[std_f == 0.0] = 1.0  
        features = (features - mean_f) / std_f 
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids]) 
        _feats = feats.reshape(-1, feats.shape[2])  
        mean = np.mean(_feats, axis=0) 
        std = np.std(_feats, axis=0)  
        std[std == 0.0] = 1.0
        return mean, std

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        missing_index = torch.cat([sample['missing_index'].unsqueeze(0) for sample in batch], axis=0)
        miss_type = [sample['miss_type'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }
    
