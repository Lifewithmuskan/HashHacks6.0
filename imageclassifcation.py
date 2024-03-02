import os
import cv2
import numpy as np
def load_images(folder):
  images = []
  labels = []
  for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    for filename in os.listdir(subfolder_path):
      img_path = os.path.join(subfolder_path, filename)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (128, 128)) 
      images.append(img)
      labels.append(subfolder)
      
    return images, labels
train_folder = "D:/ECG/train"
train_images, train_labels = load_images_from_folder(train_folder)


X_train = np.array(train_images)
y_train = np.array(train_labels)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)


X_train = X_train / 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # Assuming 4 classes: MI, History of MI, Abnormal heartbeat, Normal
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(-1, 128, 128, 1), y_train, epochs=5, batch_size=32)

def classify_ecg_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Unable to read the image. Check if the file exists and the path is correct.")
        
        img = cv2.resize(img, (128, 128))  # Resize image to 128x128 pixels
        img = np.array(img) / 255.0  # Normalize pixel values
        img = img.reshape(1, 128, 128, 1)  # Reshape image for model input
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        classes = ['MI', 'History of MI', 'Abnormal heartbeat', 'Normal']
        return classes[predicted_class]
    except Exception as e:
        print("Error:", e)
        return None



