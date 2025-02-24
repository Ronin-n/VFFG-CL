import os
import glob
import shutil
import h5py
import json
import numpy as np
from numpy.lib.function_base import extract
from tqdm import tqdm


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


def get_all_utt_id(config):
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    return all_utt_ids


def make_aligned_data(config):
    print('make_aligned_data')
    raw_L_path = os.path.join(config['feature_root'], "L", "raw_debert.h5")
    raw_L = h5py.File(raw_L_path, 'r')
    all_utt_ids = get_all_utt_id(config)
    aligned_L_path = os.path.join(config['feature_root'], "L", "deberta_large.h5")

    aligned_L_h5f = h5py.File(aligned_L_path, 'w')

    for utt_id in tqdm(all_utt_ids):
        utt_L_feat, utt_L_start, utt_L_end = \
            raw_L[utt_id]['feat'][()], raw_L[utt_id]['start'][()], raw_L[utt_id]['end'][()]

        aligned_L_h5f[utt_id] = utt_L_feat


def calc_word_aligned(word_start, word_end, frame_feats, frame_start, frame_end, default_dim=342):
    _frame_set = []
    assert word_end > word_start
    for feat, start, end in zip(frame_feats, frame_start, frame_end):
        if start == end == -1 and np.sum(frame_feats) == 0:
            break
        assert end > start, f'{start}, {end}, {frame_feats}'
        if start > word_end or end < word_start:
            continue
        else:
            _frame_set.append(feat)
    if len(_frame_set) > 0:
        _frame_set = np.array(_frame_set)
    else:
        _frame_set = np.zeros([1, default_dim])
    return np.mean(_frame_set, axis=0)

if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = '/home/txl/code/CIF-MMIN-main/data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))

    make_aligned_data(config)