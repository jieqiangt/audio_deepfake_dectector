import librosa
from model import SimpleCNN_STFT_FRAMESIZE_1024
from scripts import predict_single_audio
import torch
import argparse


def main(file_path):
    SAMPLE_RATE = 16000
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    # initialize model
    model = SimpleCNN_STFT_FRAMESIZE_1024()
    model.load_state_dict(torch.load(
        './weights/THIRD_CNN_STFT_FRAMESIZE_1024_2SEC_1_weights.pt', weights_only=True))

    y_probs = predict_single_audio(audio, model)

    return y_probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()
    main(args.file_path)
