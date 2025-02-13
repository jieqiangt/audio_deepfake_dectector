
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


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
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


class Dataset_ASVspoof2021(Dataset):
    def __init__(self, list_IDs, labels, base_dir, train=True):
        """self.list_IDs	: list of strings (each string: audio_key),
           self.labels      : dictionary (key: audio_key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        # self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.cut = 32300  # take ~4 sec audio (32300 samples)
        self.train = train

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
                   for arr in (X_log_stft, X_delta, X_delta2)]
        X = torch.FloatTensor(np.concatenate(stacked, axis=0))
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


def split_audio_dataset():

    src_dir = './data/all'
    train_dest_dir = './data/train'
    val_dest_dir = './data/val'
    test_dest_dir = './data/eval'

    if len(os.listdir(train_dest_dir)) >= 1:
        # return flac files into original folder if there are any
        print(f'found {len(os.listdir(train_dest_dir))
                       } flac file(s) in train directory...')
        print(f'transferring flac files back to train directory...')
        for file in os.listdir(train_dest_dir):
            shutil.move(f"{train_dest_dir}/{file}", f"{src_dir}/{file}")

    if len(os.listdir(val_dest_dir)) >= 1:
        # return flac files into original folder if there are any
        print(f'found {len(os.listdir(val_dest_dir))
                       } flac file(s) in val directory...')
        print(f'transferring flac files back to train directory...')
        for file in os.listdir(val_dest_dir):
            shutil.move(f"{val_dest_dir}/{file}", f"{src_dir}/{file}")

    if len(os.listdir(test_dest_dir)) >= 1:
        # return flac files into original folder if there are any
        print(f'found {len(os.listdir(test_dest_dir))
                       } flac file(s) in val directory...')
        print(f'transferring flac files back to train directory...')
        for file in os.listdir(test_dest_dir):
            shutil.move(f"{test_dest_dir}/{file}", f"{src_dir}/{file}")

    df = pd.read_csv('./data/all_labels.txt', sep=' ', header=None)
    df.rename(columns={1: 'key', 5: 'label', 7: 'group'}, inplace=True)
    df = df[['key', 'label', 'group']]

    train_total = 78000

    progress_split = 0.2
    eval_split = 0.75
    hidden_split = 0.05

    pos_split = 0.8
    neg_split = 0.2

    train_pos_progress_sampled = df[(df['group'] == 'progress') & (
        df['label'] == 'spoof')].sample(round(train_total * progress_split * pos_split))
    train_neg_progress_sampled = df[(df['group'] == 'progress') & (
        df['label'] == 'bonafide')].sample(round(train_total * progress_split * neg_split))

    train_pos_eval_sampled = df[(df['group'] == 'eval') & (
        df['label'] == 'spoof')].sample(round(train_total * eval_split * pos_split))
    train_neg_eval_sampled = df[(df['group'] == 'eval') & (
        df['label'] == 'bonafide')].sample(round(train_total * eval_split * neg_split))

    train_pos_hidden_sampled = df[(df['group'] == 'hidden') & (
        df['label'] == 'spoof')].sample(round(train_total * hidden_split * pos_split))
    train_neg_hidden_sampled = df[(df['group'] == 'hidden') & (
        df['label'] == 'bonafide')].sample(round(train_total * hidden_split * neg_split))

    train_all_sampled = pd.concat([train_pos_progress_sampled, train_neg_progress_sampled, train_pos_eval_sampled,
                                  train_neg_eval_sampled, train_pos_hidden_sampled, train_neg_hidden_sampled])[['key', 'label']]
    train_all_sampled.to_csv('./data/train_labels.txt',
                             index=False, header=False, sep=' ')

    leftover_df = df.loc[~df['key'].isin(train_all_sampled['key']), :]

    val_total = 18000
    val_progress_sampled = leftover_df[leftover_df['group'] == 'progress'].sample(
        round(val_total * progress_split))
    val_eval_sampled = leftover_df[leftover_df['group'] == 'eval'].sample(
        round(val_total * eval_split))
    val_hidden_sampled = leftover_df[leftover_df['group'] == 'hidden'].sample(
        round(val_total * hidden_split))

    val_all_sampled = pd.concat(
        [val_progress_sampled, val_eval_sampled, val_hidden_sampled])[['key', 'label']]
    val_all_sampled.to_csv('./data/val_labels.txt',
                           index=False, header=False, sep=' ')

    leftover_df = leftover_df.loc[~leftover_df['key'].isin(
        val_all_sampled['key']), :]

    test_total = 12000
    test_progress_sampled = leftover_df[leftover_df['group'] == 'progress'].sample(
        round(test_total * progress_split))
    test_eval_sampled = leftover_df[leftover_df['group'] == 'eval'].sample(
        round(test_total * eval_split))
    test_hidden_sampled = leftover_df[leftover_df['group'] == 'hidden'].sample(
        round(test_total * hidden_split))

    test_all_sampled = pd.concat(
        [test_progress_sampled, test_eval_sampled, test_hidden_sampled])[['key', 'label']]
    test_all_sampled.to_csv('./data/test_labels.txt',
                            index=False, header=False, sep=' ')

    print(
        f'transferring new split of {train_total} flac files to train data folder...')
    for key in train_all_sampled['key']:
        shutil.move(f"{src_dir}/{key}.flac", f"{train_dest_dir}/{key}.flac")

    print(
        f'transferring new split of {val_total} flac files to val data folder...')
    for key in val_all_sampled['key']:
        shutil.move(f"{src_dir}/{key}.flac", f"{val_dest_dir}/{key}.flac")

    print(
        f'transferring new split of {test_total} flac files to eval data folder...')
    for key in test_all_sampled['key']:
        shutil.move(f"{src_dir}/{key}.flac", f"{test_dest_dir}/{key}.flac")


def create_dataloader(labels_path, data_path, batch_size=12, shuffle=True, seed=42):
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    # track = config["track"]

    labels, file_list = generate_datalist(dir_meta=labels_path, with_pred=True)
    print("no. files in  dataloader:", len(file_list))

    dataset = Dataset_ASVspoof2021(list_IDs=file_list,
                                   labels=labels,
                                   base_dir=data_path,
                                   train=True)
    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=True,
                             pin_memory=True,
                             generator=gen)

    return data_loader
