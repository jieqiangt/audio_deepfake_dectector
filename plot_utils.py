import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np


def create_loss_plot(train_losses, val_losses, folder):
    
    epochs = len(train_losses)
    plt.clf()
    plt.plot(range(1,epochs+1), train_losses, label='training loss')
    plt.plot(range(1,epochs+1), val_losses, label='validation loss')
    plt.title('Training/Validation Loss per Epoch')
    plt.legend()
    plt.savefig(f'./results/{folder}/loss_plot.png')


def create_accuracy_plot(train_acc, validation_acc, folder):
    epochs = len(train_acc)
    plt.clf()
    plt.plot(range(1,epochs+1), train_acc, label='training accuracy')
    plt.plot(range(1,epochs+1), validation_acc, label='validation accuracy')
    plt.title('Training/Validation Accuracy per Epoch')
    plt.legend()
    plt.savefig(f'./results/{folder}/accuracy_plot.png')


def create_confusion_matrix(y_test, predicted, class_names, folder):
    
    arr = confusion_matrix(y_test, predicted)
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.clf()
    plt.figure(figsize=(9, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("predicted")
    plt.ylabel("ground truth")
    plt.savefig(f'./results/{folder}/confusion_matrix.png')


def create_roc_curve(fpr, tpr, roc_auc, folder):
    plt.clf()
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:0.2f}')
    plt.legend(loc='upper left')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'./results/{folder}/roc_curve.png')


def create_precision_recall_curve(recalls, precisions, folder):
    plt.clf()
    plt.title('Precision Recall Curve')
    plt.plot(recalls, precisions, 'b')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(f'./results/{folder}/p_r_curve.png')
