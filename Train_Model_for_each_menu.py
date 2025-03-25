import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image  # Add this import

# Ensure the path to images is correct
image_folder_path = 'Questionair Images'
data_question = pd.read_csv('data_from_questionaire.csv')

if os.path.exists(image_folder_path):
    print(f"The directory {image_folder_path} exists.")
else:
    raise FileNotFoundError(f"The directory {image_folder_path} does not exist.")

# Select the menu you want to train (e.g., 'burger' or 'pizza')
selected_menu = 'Burger'  # Change this to any menu you want to train on, like 'pizza'

# Filter the dataset to only include the selected menu
train_df = data_question[data_question['Menu'] == selected_menu]

# Check if the filtered dataset is empty
if train_df.empty:
    print(f"No data found for the selected menu: {selected_menu}")
    exit()  # Exit if there's no data for the selected menu

val_df = train_df.copy()  # You can adjust this to have a separate validation set if needed

# Split dataset
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Image data generators
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest')

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df, directory=image_folder_path, x_col='Image 1', y_col='Rating',  # Changed to 'Rating'
    target_size=(128, 128), class_mode=None, batch_size=32,  # Removed extra y_col
    color_mode='rgb')

val_generator = val_datagen.flow_from_dataframe(
    val_df, directory=image_folder_path, x_col='Image 1', y_col='Rating',
    target_size=(128, 128), class_mode=None, batch_size=32)

# Build CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)  # 1 output unit since we're predicting a score, not a class
])

# Compile model with mean_squared_error loss for regression
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save model
model.save('Burger_compairs.keras')

print(f"Model training complete for {selected_menu} and saved as 'food_rating_model.keras'")