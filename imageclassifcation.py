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


