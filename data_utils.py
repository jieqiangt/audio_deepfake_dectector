
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import shutil
import os

SAMPLE_RATE = 16000
FRAME_SIZE = 1024
HOP_SIZE = int(FRAME_SIZE/4)


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    elif x_len == max_len:
        return x

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2021_STFT(Dataset):
    def __init__(self, list_IDs, labels, base_dir, cut, train=True):
        """self.list_IDs	: list of strings (each string: audio_key),
           self.labels      : dictionary (key: audio_key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut  # 1 sec = 16000 samples
        self.train = train
        self.base_mean = -24.152616444707146
        self.base_std = 16.947544356542437
        self.delta_mean = 0.01327047788019662
        self.delta_std = 2.415730478109329
        self.delta2_mean = 0.002898629824448131
        self.delta2_std = 1.451467316725337

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X_raw, _ = librosa.load(
            str(f"{self.base_dir}/{key}.flac"), sr=SAMPLE_RATE)
        X_pad = pad_random(X_raw, self.cut)
        X_stft = librosa.stft(y=X_pad, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        X_log_stft = librosa.power_to_db(np.abs(X_stft)**2)
        X_delta = librosa.feature.delta(X_log_stft, width=9, order=1)
        X_delta2 = librosa.feature.delta(X_log_stft, width=9, order=2)
        stacked = [arr.reshape((1, X_log_stft.shape[0], X_log_stft.shape[1]))
                   for arr in ((X_log_stft - self.base_mean) / self.base_std, (X_delta - self.delta_mean) / self.delta_std, (X_delta2 - self.delta2_mean) / self.delta2_std)]
        X = torch.FloatTensor(np.concatenate(stacked, axis=0))

        if self.train:
            y = self.labels[key]
            return X, y

        return X, key


class Dataset_ASVspoof2021_Raw(Dataset):
    def __init__(self, list_IDs, labels, base_dir, cut, train=True):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.train = train
        self.cut = cut  # 1 second = 16000 samples

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = librosa.load(str(f"{self.base_dir}/{key}.flac"), sr=SAMPLE_RATE)
        X_pad = pad_random(X, self.cut)
        X = torch.Tensor(X_pad)

        if self.train:
            y = self.labels[key]
            return X, y

        return X, key


def tally_correct_preds(y_probs, y_truth, correct_in_epoch):

    y_preds = np.argmax(y_probs, axis=1)
    correct_in_epoch += sum(y_preds == y_truth)

    return correct_in_epoch


def generate_datalist(dir_meta, with_pred=True):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if with_pred:
        for line in l_meta:
            line_list = line.strip().split(" ")
            key = line_list[0]
            label = line_list[1]
            file_list.append(key)
            d_meta[key] = 1 if label == "spoof" else 0
        return d_meta, file_list

    for line in l_meta:
        line_list = line.strip().split(" ")
        key = line_list[0]
        label = line_list[1]
        # key = line.strip()
        file_list.append(key)
    return file_list


def split_audio_dataset(train_labels_path='./data/train_labels.txt', val_labels_path='./data/val_labels.txt'):

    bonafide_src_dir = './data/all_bonafide'
    spoof_src_dir = './data/all_spoof'

    train_dest_dir = './data/train'
    val_dest_dir = './data/val'

    if train_labels_path:
        train_labels = pd.read_csv(
            train_labels_path, sep=' ', names=['key', 'label'])

        print(f'transferring train flac files back to all_bonafide directory...')
        for file in train_labels[train_labels['label'] == 'bonafide']['key']:
            shutil.move(f"{train_dest_dir}/{file}.flac",
                        f"{bonafide_src_dir}/{file}.flac")

        print(f'transferring train flac files back to all_spoof directory...')
        for file in train_labels[train_labels['label'] == 'spoof']['key']:
            shutil.move(f"{train_dest_dir}/{file}.flac",
                        f"{spoof_src_dir}/{file}.flac")

    if val_labels_path:
        val_labels = pd.read_csv(
            val_labels_path, sep=' ', names=['key', 'label'])

        print(f'transferring val flac files back to all_bonafide directory...')
        for file in val_labels[val_labels['label'] == 'bonafide']['key']:
            shutil.move(f"{val_dest_dir}/{file}.flac",
                        f"{bonafide_src_dir}/{file}.flac")

        print(f'transferring val flac files back to all_spoof directory...')
        for file in val_labels[val_labels['label'] == 'spoof']['key']:
            shutil.move(f"{val_dest_dir}/{file}.flac",
                        f"{spoof_src_dir}/{file}.flac")

    df = pd.read_csv('./data/all_labels.txt', sep=' ', header=None)
    df.rename(columns={1: 'key', 5: 'label', 7: 'group'}, inplace=True)
    df = df[['key', 'label', 'group']]

    val_bonafide_split = 0.04
    bonafide_df = df[df['label'] == 'bonafide']
    spoof_df = df[df['label'] == 'spoof']

    num_val_bonafide = round(
        bonafide_df['key'].count().item() * val_bonafide_split)

    val_bonafide_df = bonafide_df.sample(num_val_bonafide)
    train_bonafide_df = bonafide_df[~bonafide_df['key'].isin(
        val_bonafide_df['key'])]

    val_total = 18000
    spoof_to_bonafide_ratio = 3

    val_spoof_df = spoof_df.sample(val_total - num_val_bonafide)
    leftover_spoof_df = spoof_df[~spoof_df['key'].isin(
        val_spoof_df['key'])]
    num_spoof_train = train_bonafide_df.shape[0] * spoof_to_bonafide_ratio

    train_spoof_df = leftover_spoof_df.sample(num_spoof_train)

    print(
        f'transferring new split of {train_bonafide_df.shape[0]} bonafide flac files to train data folder...')
    for key in train_bonafide_df['key']:
        shutil.move(f"{bonafide_src_dir}/{key}.flac",
                    f"{train_dest_dir}/{key}.flac")

    print(
        f'transferring new split of {train_spoof_df.shape[0]} spoof flac files to train data folder...')
    for key in train_spoof_df['key']:
        shutil.move(f"{spoof_src_dir}/{key}.flac",
                    f"{train_dest_dir}/{key}.flac")

    print(
        f'transferring new split of {val_bonafide_df.shape[0]} bonafide flac files to val data folder...')
    for key in val_bonafide_df['key']:
        shutil.move(f"{bonafide_src_dir}/{key}.flac",
                    f"{val_dest_dir}/{key}.flac")

    print(
        f'transferring new split of {val_spoof_df.shape[0]} spoof flac files to val data folder...')
    for key in val_spoof_df['key']:
        shutil.move(f"{spoof_src_dir}/{key}.flac",
                    f"{val_dest_dir}/{key}.flac")

    train_all_sampled = pd.concat([train_spoof_df, train_bonafide_df])[
        ['key', 'label']]
    train_all_sampled.to_csv('./data/train_labels.txt',
                             index=False, header=False, sep=' ')

    print(f"Total train: {train_all_sampled.shape[0]}")
    val_all_sampled = pd.concat(
        [val_spoof_df, val_bonafide_df])[['key', 'label']]
    val_all_sampled.to_csv('./data/val_labels.txt',
                           index=False, header=False, sep=' ')
    print(f'Total val: {val_all_sampled.shape[0]}')


def create_dataloader(dataset_class, labels_path, data_path, cut, batch_size=12, shuffle=True, seed=42):

    labels, file_list = generate_datalist(dir_meta=labels_path, with_pred=True)
    print("no. files in  dataloader:", len(file_list))

    dataset = dataset_class(list_IDs=file_list,
                            labels=labels,
                            base_dir=data_path,
                            train=True,
                            cut=cut)
    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=True,
                             pin_memory=True,
                             generator=gen)

    return data_loader


def pad_audio(x, max_len=32000):
    x_len = x.shape[0]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def split_audio(audio, start_range, sample_size, max_audio_size):

    end_range = start_range + sample_size
    if end_range > max_audio_size:
        ending_audio = audio[start_range:max_audio_size]
        split_audio = pad_audio(ending_audio)
        return split_audio

    split_audio = audio[start_range:end_range]
    return split_audio


def preprocess_audio_for_cnn(X_raw):

    base_mean = -24.152616444707146
    base_std = 16.947544356542437
    delta_mean = 0.01327047788019662
    delta_std = 2.415730478109329
    delta2_mean = 0.002898629824448131
    delta2_std = 1.451467316725337
    FRAME_SIZE = 1024
    HOP_SIZE = int(FRAME_SIZE/4)

    X_stft = librosa.stft(y=X_raw, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    X_log_stft = librosa.power_to_db(np.abs(X_stft)**2)
    X_delta = librosa.feature.delta(X_log_stft, width=9, order=1)
    X_delta2 = librosa.feature.delta(X_log_stft, width=9, order=2)
    stacked = [arr.reshape((1, X_log_stft.shape[0], X_log_stft.shape[1]))
               for arr in ((X_log_stft - base_mean) / base_std, (X_delta - delta_mean) / delta_std, (X_delta2 - delta2_mean) / delta2_std)]
    X = torch.FloatTensor(np.concatenate(stacked, axis=0))
    X = torch.unsqueeze(X, 0)

    return X
