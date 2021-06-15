import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def display_valid_data(model_dir, epochs):

    history_path = os.path.join(model_dir, 'history.csv')
    history = pd.read_csv(history_path, encoding = 'UTF8')
    print(history)

    acc = history['acc']
    val_acc = history['val_acc']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description='説明')

    parser.add_argument('-m', '--modeldir', default="/mnt/model", help='model directory')
    parser.add_argument('-e', '--epochs', default=15, help='epochs')

    args = parser.parse_args()

    display_valid_data(model_dir=args.modeldir, epochs=int(args.epochs))

if __name__ == "__main__":
    main()