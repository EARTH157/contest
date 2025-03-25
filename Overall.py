import os
import csv
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tkinter import Tk, Button, Label, filedialog, Toplevel
from PIL import Image, ImageTk

# Load classification model
classify_model = load_model('food_classify.keras')
menu_labels = ['Burger', 'Dessert', 'Pizza', 'Ramen', 'Sushi']  # Adjust labels if needed

# Load pairwise models
pair_models = {menu: load_model(f'food_pair_{menu}.keras') for menu in menu_labels}

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    prediction = classify_model.predict(np.expand_dims(img, axis=0))
    return menu_labels[np.argmax(prediction)]

def compare_images(model, img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1, (224, 224)) / 255.0
    img2 = cv2.resize(img2, (224, 224)) / 255.0
    prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    return '1' if prediction[0][0] > 0.5 else '2'

def show_image_pairs(image_folder):
    # Sort files in the folder to ensure correct order
    image_pairs = [
        entry.path for entry in os.scandir(image_folder)
        if entry.is_file() and entry.name.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    results = []
    index = [0]

    def update_images():
        img1_path, img2_path = image_pairs[index[0]], image_pairs[index[0] + 1]
        menu1, menu2 = classify_image(img1_path), classify_image(img2_path)
        winner = compare_images(pair_models[menu1], img1_path, img2_path)
        results.append([os.path.basename(img1_path), os.path.basename(img2_path), winner])

        # Show images in UI
        img1 = Image.open(img1_path).resize((300, 300))
        img2 = Image.open(img2_path).resize((300, 300))
        img1_tk = ImageTk.PhotoImage(img1)
        img2_tk = ImageTk.PhotoImage(img2)
        img1_label.config(image=img1_tk)
        img1_label.image = img1_tk
        img2_label.config(image=img2_tk)
        img2_label.image = img2_tk
        winner_label.config(text=f"Winner: {winner}")

    def next_pair():
        if index[0] + 2 < len(image_pairs):
            index[0] += 2
            update_images()

    def prev_pair():
        if index[0] - 2 >= 0:
            index[0] -= 2
            update_images()

    def finish():
        # Sort results according to the original order of image_pairs
        with open('test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image 1', 'Image 2', 'Winner'])
            writer.writerows(results)  # No need to sort here, just use results as they were added in order
        window.quit()  # Quit the tkinter window

    # Setup UI
    window = Toplevel()
    window.title("Image Comparison")

    img1_label = Label(window)
    img1_label.grid(row=0, column=0, padx=10, pady=10)
    img2_label = Label(window)
    img2_label.grid(row=0, column=1, padx=10, pady=10)
    winner_label = Label(window, text="", font=("Arial", 16))
    winner_label.grid(row=1, column=0, columnspan=2, pady=10)

    prev_button = Button(window, text="Previous", command=prev_pair)
    prev_button.grid(row=2, column=0, pady=10)
    next_button = Button(window, text="Next", command=next_pair)
    next_button.grid(row=2, column=1, pady=10)
    finish_button = Button(window, text="Finish", command=finish)
    finish_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Handle closing of window
    window.protocol("WM_DELETE_WINDOW", finish)

    update_images()

    window.mainloop()

def main():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Image Pairs")
    if folder_path:
        show_image_pairs(folder_path)
        print("Results saved to test.csv")

if __name__ == "__main__":
    main()