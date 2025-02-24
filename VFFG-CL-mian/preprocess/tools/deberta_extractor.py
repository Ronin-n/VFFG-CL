import os
import glob
import shutil
import h5py
import json
import numpy as np
from numpy.lib.function_base import extract
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch

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


def calc_real_time(frm):
    frm = int(frm)
    return frm / 100


def read_align_file(file):
    lines = open(file).readlines()[1:-1]
    ans = []
    for line in lines:
        line = line.strip()
        sfrm, efem, _, word = line.split()
        st = calc_real_time(sfrm)
        et = calc_real_time(efem)
        if word.startswith('<') and word.endswith('>'):
            continue
        if word.startswith('++'):
            word.replace('+', '')

        if '(' in word:
            word = word[:word.find('(')].lower()
        else:
            word = word.lower()

        if len(word) == 0:
            continue
        one_record = {
            'start_time': st,
            'end_time': et,
            'word': word,
        }
        ans.append(one_record)

    return ans

class DeBertaExtractor(object):
    def __init__(self, cuda=False, cuda_num=None):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('/home/premodel/deberta-v2-xlarge/')
        self.model = DebertaV2Model.from_pretrained('/home/premodel/deberta-v2-xlarge/')
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False

    def tokenize(self, word_lst):
        word_lst = ['[CLS]'] + word_lst + ['[SEP]']
        word_idx = []
        ids = []
        for idx, word in enumerate(word_lst):
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            token_ids = self.tokenizer.convert_tokens_to_ids(ws)
            ids.extend(token_ids)
            if word not in ['[CLS]', '[SEP]']:
                word_idx += [idx - 1] * len(token_ids)
        return ids, word_idx

    def get_embd(self, token_ids):
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        if self.cuda:
            token_ids = token_ids.to(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(token_ids)

        sequence_output = outputs.last_hidden_state  # 获取最后隐藏状态

        return sequence_output

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(input_ids)

            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output


def make_all_debert(config):
    extractor = DeBertaExtractor(cuda=True, cuda_num=1)
    word_info_dir = os.path.join(config['data_root'], 'Session{}/sentences/ForcedAlignment/{}')
    all_utt_ids = get_all_utt_id(config)
    feat_save_path = os.path.join(config['feature_root'], "L", "raw_debert.h5")
    h5f = h5py.File(feat_save_path, 'w')
    for utt_id in tqdm(all_utt_ids):
        # print("UTT_ID:", utt_id)
        session_id = int(utt_id[4])
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        if utt_id != 'Ses03M_impro03_M001':
            word_info_path = os.path.join(word_info_dir.format(session_id, dialog_id), utt_id + '.wdseg')
            word_infos = read_align_file(word_info_path)
        if utt_id == 'Ses03M_impro03_M001':
            word_lst = ['Esmeralda', 'guess', 'what']
            utt_start = [0.6, 2.06, 2.34]
            utt_end = [1.24, 2.24, 2.92]
            token_ids, word_idxs = extractor.tokenize(word_lst)
        else:
            word_lst = [x["word"] for x in word_infos]
            # print(f"word_lst: {word_lst}")
            token_ids, word_idxs = extractor.tokenize(word_lst)
            utt_start = [word_infos[i]['start_time'] for i in word_idxs]
            utt_end = [word_infos[i]['end_time'] for i in word_idxs]
            # print("utt_start:", utt_start)
            # print("utt_end:", utt_end)
        utt_feats = extractor.get_embd(token_ids)
        utt_feats = utt_feats.squeeze(0).cpu().numpy()[1:-1, :]
        assert utt_feats.shape[0] == len(utt_end)
        utt_group = h5f.create_group(utt_id)
        utt_group['feat'] = utt_feats
        utt_group['start'] = utt_start
        utt_group['end'] = utt_end

if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = '/home/txl/code/CIF-MMIN-main/data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))

    make_all_debert(config)
    # # 打印特征的形状和内容
    # print(f"word_lst: {word_lst}")
    # print("word_idxs shape:", len(word_idxs))
    # print("word_idxs:", word_idxs)
    # print("token_ids shape:", len(token_ids))
    # print("token_ids:", token_ids)
    # print("utt_start:", utt_start)
    # print("utt_end:", utt_end)
    # print("utt_feats shape:", utt_feats.shape)
    # print("Extracted utt_feats:", utt_feats)
    # print()  # 空行分隔不同文本的输出

