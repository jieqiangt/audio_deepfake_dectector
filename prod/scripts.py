from data_utils import split_audio, preprocess_audio_for_cnn
import math
import numpy as np
import torch


def predict_single_audio(audio, model):

    SAMPLE_RATE = 16000
    # split audio into ~2s chunks
    sample_size = 32300
    max_audio_size = audio.shape[0]
    audio_splits = []
    for start_range in range(0, max_audio_size, sample_size):
        audio_split = split_audio(
            audio, start_range, sample_size, max_audio_size)
        audio_splits.append(audio_split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("GPU not detected!")

    model = model.to(device)

    y_probs = []
    for X_raw in audio_splits:

        X = preprocess_audio_for_cnn(X_raw)
        X = X.to(device)
        y = model(X)
        y_probs.extend(np.repeat(y.detach().cpu().numpy()[0][1], 2).tolist())

    num_secs = math.ceil(audio.shape[0]/SAMPLE_RATE)
    y_probs = np.array(y_probs[0:num_secs])

    return y_probs
