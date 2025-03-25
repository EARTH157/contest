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

# Split dataset
train_df, val_df = train_test_split(data_question, test_size=0.2, random_state=42)

# Image data generators
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest')

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df, directory=image_folder_path, x_col='Image 1', y_col='Menu',
    target_size=(128, 128), class_mode='categorical', batch_size=32)

val_generator = val_datagen.flow_from_dataframe(
    val_df, directory=image_folder_path, x_col='Image 1', y_col='Menu',
    target_size=(128, 128), class_mode='categorical', batch_size=32)

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
    keras.layers.Dense(5, activation='softmax')  # 5 classes: burger, dessert, pizza, ramen, sushi
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save model
model.save('food_classifier_model.keras')

print("Model training complete and saved as 'food_classifier_model.keras'")