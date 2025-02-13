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
        y_probs.extend(np.repeat(y.detach().cpu().tolist()[0][1], 2).tolist())

    num_secs = math.ceil(audio.shape[0]/SAMPLE_RATE)
    y_probs = y_probs[0:num_secs]

    # threshold to determine positive label
    threshold = 0.6

    # get labels based on threshold
    labels = [1 if score >= threshold else 0 for score in y_probs]

    # get agg label for whole audio
    # logic: check if 3 consecutive seconds is labelled as 1, if it is the whole audio is declared as fake
    agg_label = 0
    for i in range(len(labels)-2):
        if labels[i] == 1 and labels[i+1] == 1 and labels[i+2] == 1:
            agg_label = 1
            break

    # calculate confidence for the whole audio
    if agg_label == 1:
        # if it is declared fake, get all probabilities above threshold and average them
        probs = list(filter(lambda score: score >= threshold, y_probs))
        confidence = np.average(probs).item()
    else:
        # if it is declared fake, get all probabilities above threshold and average them
        probs = list(filter(lambda score: score <= threshold, y_probs))
        confidence = np.average(probs).item()

    y_probs_ts = list(zip(range(0, num_secs), y_probs))

    return (y_probs_ts, confidence)
