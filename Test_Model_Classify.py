import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tkinter import Frame, Tk, Button, Label, Canvas, PhotoImage
from PIL import Image, ImageTk

# Load the trained model
model = keras.models.load_model('food_classify.keras')

# Dictionary for class labels mapping
class_labels = {
    0: 'burger',
    1: 'dessert',
    2: 'pizza',
    3: 'ramen',
    4: 'sushi'
}

# Function to preprocess image and make prediction
def predict_image(img_path):
    # Load image with target size (128x128 as used in training)
    img = image.load_img(img_path, target_size=(128, 128))

    # Convert image to numpy array and rescale
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class, predictions[0][predicted_class_idx]

# Function to get all image paths in a directory
def get_image_paths(directory):
    return (glob(os.path.join(directory, '*.jpg')) +
            glob(os.path.join(directory, '*.jpeg')) +
            glob(os.path.join(directory, '*.png')))

# GUI Setup
class ImageClassifierApp:
    def __init__(self, root, image_directory):
        self.root = root
        self.root.title("Food Image Classifier")
        
        self.image_paths = get_image_paths(image_directory)
        self.current_index = 0
        
        # Canvas to display image
        self.canvas = Canvas(root, width=300, height=300)
        self.canvas.pack()
        
        # Label to show prediction
        self.prediction_label = Label(root, text="", font=("Arial", 14))
        self.prediction_label.pack()
        
        # Buttons
        btn_frame = Frame(root)
        btn_frame.pack()
        Button(btn_frame, text="Previous", command=self.display_previous_image).pack(side="left", padx=10)
        Button(btn_frame, text="Next", command=self.display_next_image).pack(side="right", padx=10)
        
        # Display first image
        self.display_image()

    def display_image(self):
        img_path = self.image_paths[self.current_index]
        predicted_class, confidence = predict_image(img_path)
        
        # Open and resize the image for display
        img = Image.open(img_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        
        # Update canvas and prediction
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk  # Keep a reference
        self.prediction_label.config(text=f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")

    def display_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.display_image()

    def display_previous_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.display_image()

# Run the app
if __name__ == "__main__":
    image_directory = r'D:\earth\Documents\รอบแก้ตัว 26\Test Images'
    root = Tk()
    app = ImageClassifierApp(root, image_directory)
    root.mainloop()