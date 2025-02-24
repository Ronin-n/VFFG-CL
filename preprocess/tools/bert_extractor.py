import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertExtractor(object):
    def __init__(self, cuda=False, cuda_num=None):
        self.tokenizer = BertTokenizer.from_pretrained('/home/premodel/bert-base-cased/')
        self.model = BertModel.from_pretrained('/home/premodel/bert-base-cased/')
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

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output



