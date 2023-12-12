import os
import pyinotify
import torch
import tensorflow as tf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import io
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

KEYDIR = os.path.join(os.getenv("HOME"), "keydir")
HOMES = '/home'
AUTHORIZED_KEYS = '.ssh/authorized_keys'

class MLModel:
    def __init__(self):
        # Initialize your PyTorch and TensorFlow (Keras) models here
        self.torch_model = None  # Replace with your PyTorch model
        self.keras_model = None  # Replace with your TensorFlow (Keras) model

    def process_image(self, image_path):
        # Perform PyTorch image processing
        img = Image.open(image_path)
        img_tensor = torch.from_numpy(np.array(img))
        # Perform image processing tasks using PyTorch

        # Perform TensorFlow (Keras) inference
        img_array = cv2.imread(image_path)
        img_array = cv2.resize(img_array, (224, 224))  # Adjust size based on your model input size
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        prediction = self.keras_model.predict(img_array)

        return img_tensor, prediction

class UpdateKeys(FileSystemEventHandler):
    def __init__(self, ml_model):
        super().__init__()
        self.ml_model = ml_model

    def on_modified(self, event):
        if event.is_directory:
            return

        userid = os.path.basename(event.src_path).split("@")[0]
        userdir = os.path.join(HOMES, userid)
        if not os.path.exists(userdir):
            return

        new_key = ''
        for root, _, files in os.walk(KEYDIR):
            for f in files:
                if f.startswith(userid):
                    key_path = os.path.join(root, f)
                    with open(key_path, 'r') as key_file:
                        new_key += key_file.read()

                    # Perform ML tasks when a key file is modified
                    img_tensor, prediction = self.ml_model.process_image(key_path)

        combined_key_file = os.path.join(userdir, AUTHORIZED_KEYS)
        with open(combined_key_file, 'w') as combined_file:
            combined_file.write(new_key)

if __name__ == "__main__":
    ml_model = MLModel()
    event_handler = UpdateKeys(ml_model)
    observer = Observer()
    observer.schedule(event_handler, KEYDIR, recursive=True)
    observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
