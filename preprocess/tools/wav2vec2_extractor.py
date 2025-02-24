import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import torchaudio


class Wav2vecExtractor(object):
    ''' 抽取wav2vec特征, 输入音频路径, 输出npy数组, 每帧768d '''

    def __init__(self):
        ''' Extract ComparE feature '''
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("/home/premodel/wav2vec2-base-960h/")
        self.model_SC = Wav2Vec2ForSequenceClassification.from_pretrained("/home/premodel/wav2vec2-base-960h/")
        self.model = Wav2Vec2Model.from_pretrained("/home/premodel/wav2vec2-base-960h/")
        self.model.eval()  # 设置为评估模式

    def forward(self, x):
        return self.model(x).last_hidden_state

    def feat_extractor(self, x, sampling_rate):
        feat_a = self.feature_extractor(x, sampling_rate=sampling_rate, return_tensors="pt")
        return feat_a


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


def make_all_wav2vec(config):
    extractor = Wav2vecExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')

    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))

    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'A', 'wav2vec2.h5'), 'w')

    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        wav_path = os.path.join(config['data_root'], f'Session{ses_id}', 'sentences', 'wav', f'{dialog_id}',
                                f'{utt_id}.wav')

        try:
            # Load audio file
            if not os.path.exists(wav_path):
                print(f"Audio file not found: {wav_path}")
                continue

            waveform, sample_rate = torchaudio.load(wav_path)

            # 如果需要，将 waveform 转换为 numpy 数组
            if waveform.ndim > 1:
                waveform = waveform[0]  # 只保留一个通道
            waveform_numpy = waveform.squeeze().numpy()  # 移除多余的维度并转换为 numpy 数组

            # 打印结果
            print("Waveform shape:", waveform_numpy.shape)
            print("Sample rate:", sample_rate)

            # Process the waveform using the processor
            input_a = extractor.feat_extractor(waveform_numpy, sample_rate)  # Use the numpy array as input

            # Forward sample through model
            feat = extractor.forward(input_a)

            # 转换为二维特征 (seq_len, 768)
            if feat.ndim > 2:
                feat_2d = feat.squeeze(0).detach().numpy()  # 使用 squeeze 去掉第一维
            else:
                feat_2d = feat.detach().numpy()  # 根据需要调整

            # Store features in HDF5 file
            all_h5f[utt_id] = feat_2d  # 使用二维特征

        except Exception as e:
            print(f"Error processing {utt_id}: {e}")

    all_h5f.close()  # 关闭 HDF5 文件

def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))

if __name__ == "__main__":
    config_path = '/home/txl/code/CIF-MMIN-main/data/config/IEMOCAP_config.json'
    config = json.load(open(config_path))
    make_all_wav2vec(config)
    # normlize_on_trn(config, os.path.join(config['feature_root'], 'A', 'wav2vec2_new.h5'),
    #                 # os.path.join(config['feature_root'], 'A', 'wav2vec2_new_mean_std.h5'))