import os
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F
import torch.utils.data
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

import random
import torchaudio
import torch

import numpy as np


def load_wav(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(str(full_path))
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
    # return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def mel_normalize(S, clip_val=1e-5):
    S = (S - torch.log(torch.Tensor([clip_val]))) * 1.0 / (0 - torch.log(torch.Tensor([clip_val])))
    return S


#
#
def mel_denormalize(S, clip_val=1e-5):
    S = S * (0 - torch.log(torch.Tensor([clip_val])).cuda()) + torch.log(torch.Tensor([clip_val])).cuda()
    return S


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # print('min value is ', torch.min(y))
    # print('max value is ', torch.max(y))
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def get_dataset_filelist(wavs_dir):
    file_list = []
    for figure in os.listdir(wavs_dir):
        figure_dir = os.path.join(wavs_dir, figure)
        for file_name in os.listdir(figure_dir):
            file_list.append(os.path.join(figure_dir, file_name))
    return file_list


def get_test_dataset_filelist2(wavs_dir):
    source_dir = os.path.join(wavs_dir,'src3000_22050')
    target_dir = os.path.join(wavs_dir,'tar3000_22050')
    # source_dir = os.path.join(wavs_dir, 'src280_22050')
    # target_dir = os.path.join(wavs_dir, 'tar280_22050')

    # source_dir = wavs_dir
    # target_dir = wavs_dir
    source_file_list = os.listdir(source_dir)
    target_file_list = os.listdir(target_dir)
    source_file_list.sort()
    target_file_list.sort()
    choice_list = []

    r = random.random
    random.seed(56)
    random.shuffle(source_file_list, random=r)
    source_file_list = source_file_list[0:500]
    target_file_list = target_file_list[0:500]

    for index,source_file in enumerate(source_file_list):
        source_file_path = os.path.join(source_dir,source_file)
        target_file = target_file_list[index]

        target_file_path = os.path.join(target_dir, target_file)
        choice_list.append([source_file_path, target_file_path])
    return choice_list


def get_test_dataset_filelist(wavs_dir):
    file_list = [os.path.join(wavs_dir, file_name) for file_name in os.listdir(wavs_dir)]
    choice_list = []
    for i in range(60):
        source_file = random.choice(file_list)
        target_file = random.choice(file_list)
        choice_list.append([source_file, target_file])
    return choice_list


def get_infer_test_dataset_filelist(wavs_dir):
    src_path_list = []
    tar_path_list = []
    for file_name in os.listdir(wavs_dir):
        if 'src' in file_name:
            src_path_list.append(os.path.join(wavs_dir, file_name))
        else:
            tar_path_list.append(os.path.join(wavs_dir, file_name))
    src_path_list.sort()
    tar_path_list.sort()
    assert len(src_path_list) == len(tar_path_list)
    choice_list = []
    for i in range(len(src_path_list)):
        choice_list.append([src_path_list[i], tar_path_list[i]])
    return choice_list


def get_infer_test_dataset_filelist(wavs_dir):
    figure_list = os.listdir(wavs_dir)
    choice_list = []
    for i in range(1000):
        source_figure = random.choice(figure_list)
        source_figure_dir = os.path.join(wavs_dir, source_figure)
        source_file_list = [os.path.join(source_figure_dir, file_name) for file_name in os.listdir(source_figure_dir)]
        source_file = random.choice(source_file_list)

        target_figure = random.choice(figure_list)
        while target_figure==source_figure:
            target_figure = random.choice(figure_list)
        target_figure_dir = os.path.join(wavs_dir, target_figure)
        target_file_list = [os.path.join(target_figure_dir, file_name) for file_name in os.listdir(target_figure_dir)]
        target_file = random.choice(target_file_list)

        choice_list.append([source_file, target_file])

    return choice_list

def get_infer_test_dataset_filelist2(wavs_dir):
    file_list = os.listdir(wavs_dir)
    choice_list = []
    for i in range(1000):
        source_name = random.choice(file_list)
        source_file = os.path.join(wavs_dir, source_name)
        target_name = random.choice(file_list)
        target_file = os.path.join(wavs_dir, target_name)

        choice_list.append([source_file, target_file])
    return choice_list

def get_infer_test_dataset_filelist3(src_wavs_dir,tar_wavs_dir):
    src_file_list = os.listdir(src_wavs_dir)
    tar_file_list = os.listdir(tar_wavs_dir)

    r = random.random
    random.seed(56)
    random.shuffle(tar_file_list, random=r)

    choice_list = []
    for index in range(min(len(src_file_list),len(tar_file_list))):
        source_file = os.path.join(src_wavs_dir, src_file_list[index])
        target_file = os.path.join(tar_wavs_dir, tar_file_list[index])
        choice_list.append([source_file, target_file])
    return choice_list

def get_visual_test_dataset_filelist(wavs_dir):
    source_dir = os.path.join(wavs_dir, 'source')
    target_dir = os.path.join(wavs_dir, 'target')
    source_file_list = os.listdir(source_dir)
    target_file_list = os.listdir(target_dir)

    choice_list = []
    for source_file in source_file_list:
        source_file_path = os.path.join(source_dir, source_file)
        target_file = random.choice(target_file_list)
        target_file_path = os.path.join(target_dir, target_file)
        choice_list.append([source_file_path, target_file_path])
    return choice_list

def fixed_length(mel, segment_len=128):
    if mel.shape[0] < segment_len:
        len_pad = segment_len - mel.shape[0]
        mel = F.pad(mel, (0, 0, 0, len_pad), 'constant')
        assert mel.shape[0] == segment_len
    elif mel.shape[0] > segment_len:
        left = np.random.randint(mel.shape[0] - segment_len)
        mel = mel[left:left + segment_len, :]
    return mel

# def fixed_length(mel, segment_len=128):
#     if mel.shape[0] < segment_len:
#         len_pad = segment_len - mel.shape[0]
#         mel = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
#         assert mel.shape[0] == segment_len
#     elif mel.shape[0] > segment_len:
#         left = np.random.randint(mel.shape[0] - segment_len)
#         mel = mel[left:left + segment_len, :]
#     return mel



def pad_mul_segment(mel,segment_len=128):
    pad_len = mel.shape[0]%segment_len
    if pad_len==0:
        return mel
    pad_len = segment_len-pad_len

    long_mel_list = []
    num = pad_len//mel.shape[0]+1
    for i in range(num):
        long_mel_list.append(mel)
    long_mel = torch.cat(long_mel_list,0)
    pad_mel = long_mel[:pad_len, :]
    mul_mel = torch.cat([mel,pad_mel],0)
    assert mul_mel.shape[0]%segment_len==0
    return mul_mel

def w2v_load_wav(audio_path, sample_rate: int, trim: bool = False) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]
    return wav

# any2one_path = "/home/gyw/workspace/VC_GAN/SingleVC_GAN/patch/any2one_gan_patch/output_equal_encoder_lr/2021_11_25_06_22_16/model/checkpoint-2300.pt"
# any2one_path = "/home/gyw/workspace/VC_GAN/SingleVC_GAN/patch/any2one_gan_patch/output_equal_encoder_lr/bottle256_3999_077/model/checkpoint-2300.pt"
# any2one_path = "/home/gyw/workspace/program/VC/MediumVC/Any2Any_new_spk/model/checkpoint-3000.pt"
#any2one_path = "/home/gyw/workspace/VC_GAN/SingleVC_GAN/patch/any2one_gan_patch/output_equal_encoder_lr/2021_11_28_21_20_17/model/checkpoint-1700.pt"


# any2one_path = "/home/gyw/workspace/program/VC_GAN/LightVC/checkpoint-3999.pt"
class Test_MelDataset(torch.utils.data.Dataset):
    def __init__(self, test_files, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, device=None):
        self.audio_files = test_files
        self.sampling_rate = sampling_rate

        self.n_fft = n_fft
        # self.num_mels = num_mels
        self.src_num_mels = 80
        self.tar_num_mels = 80
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.device = device

        # self.any2one_model = get_any2one_encoder(any2one_path)

    def __getitem__(self, index):
        src_filename, tar_filename = self.audio_files[index]
        src_file_label = os.path.basename(src_filename).split(".")[0]
        tar_file_label = os.path.basename(tar_filename).split(".")[0]
        convert_label = src_file_label + 'TO' + tar_file_label
        ####### len
        src_audio, src_sampling_rate = load_wav(src_filename)
        ### split
        src_audio = normalize(src_audio) * 0.95
        src_audio = torch.FloatTensor(src_audio)
        src_audio = src_audio.unsqueeze(0)
        src_mel = mel_spectrogram(src_audio, self.n_fft, self.src_num_mels, self.sampling_rate, self.hop_size,
                                  self.win_size, self.fmin, self.fmax, center=False)

        src_mel = src_mel.squeeze(0).transpose(0, 1)
        src_mel = mel_normalize(src_mel)

        # src_mel = src_mel.unsqueeze(0)
        # fake_mel = self.any2one_model(src_mel)
        # fake_mel = torch.clamp(fake_mel, min=0, max=1)
        # fake_mel = fake_mel.squeeze(0)  # len dim
        fake_mel = src_mel


        ###### tar
        tar_tensor, sample_rate = load_wav(tar_filename)
        clip_audio, _ = librosa.effects.trim(tar_tensor, top_db=20)
        clip_audio = normalize(clip_audio) * 0.95
        clip_audio = torch.FloatTensor(clip_audio)
        clip_audio = clip_audio.unsqueeze(0)
        clip_mel = mel_spectrogram(clip_audio, self.n_fft, self.tar_num_mels, self.sampling_rate, self.hop_size,self.win_size, self.fmin, self.fmax, center=False)# 1,80,167
        clip_mel = clip_mel.squeeze(0).transpose(0, 1)

        clip_mel = mel_normalize(clip_mel)
        clip_mel = pad_mul_segment(clip_mel,128)
        # clip_mel = pad_segment(clip_mel)
        # clip_mel = fixed_length(clip_mel)


        # return src_mel, clip_mel, convert_label
        return fake_mel, clip_mel, convert_label

    def __len__(self):
        return len(self.audio_files)



