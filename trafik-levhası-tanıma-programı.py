import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint

inputBasePath = r"    " #https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign    bu linkten veri setini indirip yolunu belirtin
trainingFolder = 'DATA'
labels_csv_path = os.path.join(inputBasePath, "labels.csv")
model_path = os.path.join(inputBasePath, "traffic_sign_model.keras")
epoch_info_path = os.path.join(inputBasePath, "epoch_info.txt")

def fetch_images(data, folder):
    classFolders = os.listdir(os.path.join(inputBasePath, folder))
    for classValue in classFolders:
        classPath = os.path.join(inputBasePath, folder, classValue)
        if os.path.isdir(classPath):
            for trafficSignal in os.listdir(classPath):
                imgTrafficSignal = cv2.imread(os.path.join(classPath, trafficSignal), cv2.IMREAD_GRAYSCALE)
                imgTrafficSignal = cv2.resize(imgTrafficSignal, (90, 90))
                data.append((imgTrafficSignal, int(classValue)))
    return data

traffic_data = fetch_images([], trainingFolder)
traffic_data_features, traffic_data_labels = zip(*traffic_data)

training_data_features, _, training_data_labels, _ = train_test_split(
    traffic_data_features, traffic_data_labels, test_size=0.1, random_state=42)

training_data_features = np.array(training_data_features)
training_data_labels = np.array(training_data_labels)

def convolution_model():
    cnnModel = Sequential([
        Conv2D(16, (3, 3), padding="same", input_shape=(90, 90, 1), activation='relu'),
        MaxPool2D((2, 2), padding="same"),
        Conv2D(32, (3, 3), padding="same", activation='relu'),
        MaxPool2D((2, 2), padding="same"),
        Conv2D(64, (5, 5), padding="same", activation='relu'),
        MaxPool2D((2, 2), padding="same"),
        Conv2D(128, (7, 7), padding="same", activation='relu'),
        MaxPool2D((2, 2), padding="same"),
        Flatten(),
        Dense(232, activation='relu'),
        Dense(116, activation='relu'),
        Dense(len(set(traffic_data_labels)), activation='softmax')
    ])
    return cnnModel
epochs = 30
batchSize = 15

if os.path.exists(model_path) and os.path.exists(epoch_info_path):
    with open(epoch_info_path, 'r') as f:
        model_epochs = int(f.read().strip())

    if model_epochs == epochs:
        cnnModel = load_model(model_path)
        print("Kaydedilen model yüklendi.")
    else:
        cnnModel = convolution_model()
        cnnModel.summary()
        cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        
        cnnModel.fit(np.expand_dims(training_data_features, axis=-1), training_data_labels,
                     batch_size=batchSize, epochs=epochs, callbacks=[checkpoint], validation_split=0.1)
        
        with open(epoch_info_path, 'w') as f:
            f.write(str(epochs))
        
        print("Model yeniden eğitildi ve kaydedildi.")
else:
    cnnModel = convolution_model()
    cnnModel.summary()
    cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    
    cnnModel.fit(np.expand_dims(training_data_features, axis=-1), training_data_labels,
                 batch_size=batchSize, epochs=epochs, callbacks=[checkpoint], validation_split=0.1)
    
    with open(epoch_info_path, 'w') as f:
        f.write(str(epochs))
    
    print("Model eğitildi ve kaydedildi.")

class_names = pd.read_csv(labels_csv_path, encoding='latin1')['Name'].tolist()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    gray_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (90, 90))

    prediction = cnnModel.predict(np.expand_dims(np.array([gray_resized]), axis=-1))

    predicted_label = np.argmax(prediction, axis=1)
    predicted_name = class_names[predicted_label[0]]
    confidence = prediction[0][predicted_label[0]] * 100
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    text_x = 10
    text_y = 30

    cv2.putText(frame, f"{predicted_name}: {confidence:.2f}%", (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    height, width = frame.shape[:2]
    start_row, start_col = int(height * .25), int(width * .25)
    end_row, end_col = int(height * .75), int(width * .75)
    cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)

    cv2.imshow('Trafik İşareti Tanıma', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
