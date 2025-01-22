import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn


def create_loss_plot(train_losses, val_losses, file_name):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.title('Training/Validation Loss per Epoch')
    plt.legend()
    plt.savefig(f'./results/loss_plot/{file_name}.png')

def create_accuracy_plot(train_correct,validation_correct, file_name):
    plt.plot([t/500 for t in train_correct], label='training accuracy')
    plt.plot([t/100 for t in validation_correct], label='validation accuracy')
    plt.title('Accuracy at the end of each epoch')
    plt.legend()
    plt.savefig(f'./results/accuracy_plot/{file_name}.png')
    
def create_confusion_matrix(y_test, predicted, class_names, file_name):
    
    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show()
    plt.savefig(f'./results/confusion_matrix/{file_name}.png')