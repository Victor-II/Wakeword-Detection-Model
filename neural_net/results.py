import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(history=None):
    
    history = pd.read_csv(history)
        
    plt.figure(figsize=(10, 6))
    history['train_accuracy'].plot(grid=True, marker='o', markersize=4, color='tab:orange')  
    history['val_accuracy'].plot(grid=True, xlabel='Epochs', marker='o', markersize=4, color='tab:blue')
    plt.title('Validation Accuracy vs Training Accuracy')
    plt.legend()
    plt.show()

def plot_loss(history=None):
    
    history = pd.read_csv(history)
        
    plt.figure(figsize=(10, 6))
    history['val_loss'].plot(grid=True, xlabel='Epochs', marker='o', markersize=4, color='tab:blue')
    history['train_loss'].plot(grid=True, marker='o', markersize=4, color='tab:orange')
    plt.title('Validation Loss vs Training Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    history = 'C:/Users/Victor/Desktop/crdmProiect/wakeword_detection/saved_history/history.csv'
    plot_accuracy(history)
    plot_loss(history)