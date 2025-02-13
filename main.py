import os
import torch
from torch import nn
from data_utils import create_dataloader
from scripts import train
from eval_utils import generate_metrics
from models import SimpleCNN_STFT_FRAMESIZE_1024
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


def main():

    # metadata for iteration
    model_name = 'THIRD_CNN_STFT_FRAMESIZE_1024_2SEC'
    iteration = 1
    results_folder = f'{model_name}_{iteration}'

    # hyper parameters
    batch_size = 48
    learning_rate = 0.0001
    epochs = 10
    weight_decay = 0.0001
    amsgrad = False
    betas = [0.9, 0.999]

    # check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name()
    print(f"GPU: {device_name}")
    if device_name == "cpu":
        raise ValueError("GPU not detected!")

    # set up model
    model = SimpleCNN_STFT_FRAMESIZE_1024()
    total_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)

    print(f'total_params: {total_params}')

    # set up dataloader & batch size
    train_labels_path = './data/train_labels.txt'
    train_data_path = './data/train'
    train_loader = create_dataloader(
        train_labels_path, train_data_path, batch_size)

    val_labels_path = './data/val_labels.txt'
    val_data_path = './data/val'
    val_loader = create_dataloader(
        val_labels_path, val_data_path, batch_size, shuffle=False)

    # set criterion & optimizer
    weight = torch.FloatTensor([0.8, 0.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(
    ), lr=learning_rate, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)

    # train
    train_losses, train_correct, val_losses, val_correct, elapsed_time, y_pos_probs, y_tests = train(
        model, train_loader, val_loader, criterion, optimizer, epochs, batch_size, device)

    # generate & store metrics in results folder
    if not os.path.exists(f"./results/{results_folder}"):
        os.makedirs(f"./results/{results_folder}")
    train_acc = [t/len(train_loader.dataset) * 100 for t in train_correct]
    val_acc = [t/len(val_loader.dataset) * 100 for t in val_correct]

    generate_metrics(batch_size, epochs, learning_rate, train_losses, val_losses, train_acc, val_acc,
                     y_tests, y_pos_probs, total_params, model_name, iteration, device_name, elapsed_time, results_folder)

    torch.save(model.state_dict(), f'models/{results_folder}_weights.pt')


if __name__ == "__main__":
    main()
