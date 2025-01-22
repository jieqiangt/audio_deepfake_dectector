import time
import torch
from utils import tally_correct_preds

def train(model, train_dataloader, val_dataloader, y_pred, criterion, optimizer, epochs):
    
    start_time = time.time()
    train_losses = []
    train_correct = []
    val_losses = []
    val_correct = []

    for epoch in range(epochs):
        train_correct_in_epoch = 0
        epoch+=1
        
        # Run the training batches
        for batch, (X_train, y_train) in enumerate(train_dataloader):
            batch+=1
            
            # Apply the model
            y_pred = model(X_train)
            train_loss = criterion(y_pred, y_train)
    
            train_correct_in_epoch = tally_correct_preds(y_pred, y_train, train_correct_in_epoch)
            
            # Update parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Print interim results
            if batch%1000 == 0:
                print(f'epoch: {epoch:2}  batch: {batch:4} [{10*batch:6}/50000]  loss: {train_loss.item():10.8f} accuracy: {train_correct_in_epoch.item()*100/(10*batch):7.3f}%')
            
        train_losses.append(train_loss)
        train_correct.append(train_correct_in_epoch)
        
        val_loss, val_correct_in_epoch = predict(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        val_correct.append(val_correct_in_epoch)
        
        elapsed_time = time.time() - start_time
        print(f'\n epoch: {epoch:2} Duration: {elapsed_time:.0f} seconds') # print the time elapsed  
        
    return train_losses, train_correct, val_losses, val_correct, elapsed_time

def predict(model, test_dataloader, criterion):

    # Run the testing batches
    correct_in_epoch = 0
    
    with torch.no_grad():
        for _, (X_test, y_test) in enumerate(test_dataloader):
            
            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            correct_in_epoch = tally_correct_preds(y_val, y_test, correct_in_epoch)
            
    loss = criterion(y_val, y_test)
    
    return loss, correct_in_epoch


        

            
      