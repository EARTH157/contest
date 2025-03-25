import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore
import os

# Load the trained model
model = keras.models.load_model('Burger_compairs.keras')

# Path to test images
image_folder_path = 'Questionair Images'

# Test data (you can create a CSV file for testing as mentioned above)
test_df = pd.read_csv('test_data.csv')

# Function to preprocess image before prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Loop through the test data and make predictions
for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder_path, row['Image 1'])
    
    if os.path.exists(img_path):
        # Preprocess image
        img_array = preprocess_image(img_path)
        
        # Predict rating
        prediction = model.predict(img_array)
        
        print(f"Predicted Rating for {row['Image 1']} ({row['Menu']}): {prediction[0][0]:.2f}")
    else:
        print(f"Image {row['Image 1']} not found in the directory.")