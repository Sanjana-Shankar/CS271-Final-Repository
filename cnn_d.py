
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models, optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================================
# Set up Datasets
# Load arrays with each distinct malware samples
# ==========================================================
def load_data(data_path):

    data_d = []
    labels_d = []

    count = 0
    for file in os.listdir(data_path):
        if file == ".DS_Store" or file.endswith('.npz') or file.endswith('.npy'):
            continue
        print(f"Processing dataset Family {file}")
        # path for each folders A to L
        folder_path = os.path.join(data_path, file)
        print(folder_path)
        for images in os.listdir(folder_path):
            class_label = images[0]  # Extract class label from filename
            if class_label == 'A':
                class_label = 0
            elif class_label == 'B':
                class_label = 1
            elif class_label == 'C':
                class_label = 2
            elif class_label == 'D':
                class_label = 3
            elif class_label == 'E':
                class_label = 4
            elif class_label == 'F':
                class_label = 5
            elif class_label == 'G':
                class_label = 6
            elif class_label == 'H':
                class_label = 7
            elif class_label == 'I':
                class_label = 8
            elif class_label == 'J':
                class_label = 9
            elif class_label == 'K':
                class_label = 10
            elif class_label == 'L':
                class_label = 11
            
            # # Skip non-TIFF files
            if not images.lower().endswith('.tiff'):
                continue
            
            # skip folder X
            if folder_path.endswith('X'):
                continue
            else:
                if images.lower().endswith("d.tiff"):
                    try:
                        images = os.path.join(folder_path, images)
                        img = Image.open(images)
                        img = img.convert("RGB")  # Ensure image is in RGB format
                        img = img.resize((224, 224))
                        img_array = np.array(img)
                    except Exception as e:
                        print(f"Error opening {images}: {e}")
                    
                    labels_d.append(class_label)
                    data_d.append(img_array)
                    count += 1
        print(f"Processed {count} .tiff malware images.")
    print(len(data_d))

    # Save the datasets as .npz files
    np.savez("CS271_final_data/cnn_d.npz", data_d=data_d, labels_d=labels_d)
    return data_d, labels_d

def cnn_model(data_d, labels_d):
    # ==========================================================
    # Train CNN on malware sample a to f
    # ==========================================================

    # CNN D
    X = np.array(data_d)
    y = np.array(labels_d)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalize pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    model = models.Sequential([
        layers.InputLayer(shape=(224, 224, 3)),

        layers.Conv2D(filters=32, kernel_size=(3, 3),activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3),activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=128, kernel_size=(3, 3),activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(12, activation='softmax'),
    ])

    optimizer = optimizers.Adam(learning_rate=0.001)
    epochs = 15

    # optimizer
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    checkpoint_dir = './CS271_final_data/model/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = './CS271_final_data/model/checkpoint.model_d.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    history = model.fit(X_train, y_train, epochs=epochs, 
                        batch_size=32, 
                        validation_data=(X_test, y_test),
                        callbacks=[model_checkpoint_callback])



    
    return model, history, X_test, y_test


def evaluation(model, history, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, np.argmax(y_pred, axis=1)))

def plots(history, model):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'CNN Model {model}: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'./Images/cnn_model_{model}_accuracy.png')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'CNN Model {model}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'./Images/cnn_model_{model}_loss.png')
    plt.show()

def main():
    # Load and preprocess the dataset
    data_path = Path("./CS271_final_data")
    if not os.path.exists("CS271_final_data/cnn_d.npz"):
        print("cnn_d.npz not found, loading data and creating it...")
        data_d, labels_d = load_data(data_path)
        print("Saved cnn_d.npz")
    else:
        print("Found cnn_d.npz. Loading file...")
        data = np.load("CS271_final_data/cnn_d.npz", allow_pickle=True)
        data_d = data['data_d']
        labels_d = data['labels_d']
        print("Loaded cnn_d.npz")
    
    model_d, history_d, X_test, y_test = cnn_model(data_d, labels_d)
    evaluation(model_d, history_d, X_test, y_test)
    plots(history_d, 'd')
    return model_d, history_d

if __name__ == "__main__":
    model_d, history_d = main()

