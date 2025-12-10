import os
from tkinter import Image
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def load_tiff_dataset(path="CS271_final_data", img_size=(224, 224)):
    images = []
    labels = []

    # Loop through all subfolders (A, B, C, ..., X)
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Folder path: {folder_path}")

        for filename in os.listdir(folder_path):
            if (not filename.lower().endswith(("_e.tiff")) or filename.startswith("X")):
                continue

            # Example filename: "L_883_c.tiff"
            #class_label = filename.split("_")[-1].split(".")[0]
            class_label = filename[0].upper()

            file_path = os.path.join(folder_path, filename)
            print(f"File path: {file_path}")

            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize(img_size)
                images.append(np.array(img))
                if(class_label == 'A'):
                    labels.append(0)
                elif(class_label == 'B'):
                    labels.append(1)
                elif(class_label == 'C'):
                    labels.append(2)
                elif(class_label == 'D'):
                    labels.append(3)
                elif(class_label == 'E'):
                    labels.append(4)
                elif(class_label == 'F'):
                    labels.append(5)
                elif(class_label == 'G'):
                    labels.append(6)
                elif(class_label == 'H'):
                    labels.append(7)
                elif(class_label == 'I'):
                    labels.append(8)
                elif(class_label == 'J'):
                    labels.append(9)
                elif(class_label == 'K'):
                    labels.append(10)
                elif(class_label == 'L'):
                    labels.append(11)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Convert to arrays
    images = np.array(images)
    labels = np.array(labels)

    print("Loaded images:", images.shape)
    print("Loaded labels:", labels.shape)
    print("Unique classes:", np.unique(labels))

    return images, labels


images, labels = load_tiff_dataset("CS271_final_data", (224, 224))

print("Total images loaded:", len(images))
print("Labels:", labels)

# 80:20 split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(images, labels, test_size=0.20, random_state=42, stratify=labels)

X_train_e = X_train_e.astype("float32") / 255.0
X_test_e = X_test_e.astype("float32") / 255.0

print("Train size for class a:", len(X_train_e))
print("Test size for class a:", len(X_test_e))

y_train_e = y_train_e.astype(np.int32)
y_test_e = y_test_e.astype(np.int32)

print(X_train_e.dtype, X_train_e.shape)
print(type(X_train_e[0]))
print(y_train_e[:10])
print(type(y_train_e[0]))


def train_CNN_class_e(X_train, y_train, X_test, y_test):
    
    # Create an input layer 
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3)) # 224 x 224 pixel images with a single color channel

    # CNN model building 

    model = tf.keras.Sequential([
        input_layer, # input layer
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'), # convolutional layer 
        # filter is the number of filters we want to apply
        # kernel is the size of window/filter moving over the image 
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'), # convolutional layer
        layers.MaxPooling2D(), # pooling layer
        # layers.Dropout(0.1),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'), # convolutional layer
        # layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'), # convolutional layer 
        layers.MaxPooling2D(), # pooling layer 

        layers.Flatten(), # flatten layer
        layers.Dense(64, activation = 'relu'),
        layers.Dropout(0.4),
        layers.Dense(12, activation='softmax') # output layer 
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

    loss,accuracy = model.evaluate(X_test, y_test)
    
    return accuracy, model, history

accuracy, model, history = train_CNN_class_e(X_train_e, y_train_e, X_test_e, y_test_e)
print(f"Test accuracy: {accuracy * 100:.2f}%")
model.save('cnn_class_e_model.h5')

plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('CNN: Model E Accuracy Graph')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('CNN: Model E Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()