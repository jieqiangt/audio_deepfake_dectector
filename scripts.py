import time
import torch
from data_utils import tally_correct_preds
import numpy as np


def predict(model, test_dataloader, criterion, device):

    y_pos_probs = []
    y_tests = []
    # Run the testing batches
    correct_in_epoch = 0
    with torch.no_grad():
        for _, (X_test, y_test) in enumerate(test_dataloader):

            # Apply the model
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = model(X_test)

            # Tally the number of correct predictions
            correct_in_epoch = tally_correct_preds(
                y_val.detach().cpu().numpy(), y_test.detach().cpu().numpy(), correct_in_epoch)
            y_pos_prob = y_val[np.repeat(
                [[False, True]], y_val.shape[0], axis=0)]
            y_pos_probs.extend(y_pos_prob.tolist())
            y_tests.extend(y_test.tolist())

    loss = criterion(y_val, y_test)

    return loss, correct_in_epoch, y_pos_probs, y_tests


def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, batch_size, device):

    start_time = time.time()
    train_losses = []
    train_correct = []
    val_losses = []
    val_correct = []

    for epoch in range(epochs):
        train_correct_in_epoch = 0
        model.train()
        epoch += 1
        print(f'starting epoch {epoch}...')
        # Run the training batches
        for batch, (X_train, y_train) in enumerate(train_dataloader):
            batch += 1

            # Apply the model
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(X_train)
            train_loss = criterion(y_pred, y_train)

            train_correct_in_epoch = tally_correct_preds(
                y_pred.detach().cpu().numpy(), y_train.detach().cpu().numpy(), train_correct_in_epoch)

            # Update parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f'epoch: {epoch:2}  batch: {batch} [{batch_size*batch:6}]  loss: {train_loss.item(
                ):10.8f} accuracy: {train_correct_in_epoch.item()/(batch_size*batch)*100:7.3f}% elapsed time: {(time.time() - start_time):7.3f}')

        train_losses.append(train_loss.detach().cpu().numpy().item())
        train_correct.append(
            train_correct_in_epoch.item())

        model.eval()
        val_loss, val_correct_in_epoch, y_pos_probs, y_tests = predict(
            model, val_dataloader, criterion, device)
        val_losses.append(val_loss.detach().cpu().numpy().item())
        val_correct.append(val_correct_in_epoch.item())

        elapsed_time = time.time() - start_time
        print(f'\n epoch: {epoch:2} Duration: {elapsed_time:.0f} seconds')

    total_elapsed_time = time.time() - start_time

    return train_losses, train_correct, val_losses, val_correct, total_elapsed_time, y_pos_probs, y_tests
