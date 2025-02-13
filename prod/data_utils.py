import numpy as np
import librosa
import torch


def pad_audio(x, max_len=32300):
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

    FRAME_SIZE = 1024
    HOP_SIZE = int(FRAME_SIZE/4)

    X_stft = librosa.stft(y=X_raw, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    X_log_stft = librosa.power_to_db(np.abs(X_stft)**2)
    X_delta = librosa.feature.delta(X_log_stft, width=9, order=1)
    X_delta2 = librosa.feature.delta(X_log_stft, width=9, order=2)
    stacked = [arr.reshape((1, X_log_stft.shape[0], X_log_stft.shape[1]))
               for arr in (X_log_stft, X_delta, X_delta2)]
    X = torch.FloatTensor(np.concatenate(stacked, axis=0))

    return X
