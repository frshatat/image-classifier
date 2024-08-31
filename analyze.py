from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('./image-class-model.h5')

def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255  # Normalize pixel values
    return img_array

def visualize_image(image_array):
    # Reshape the image array for visualization
    img = image_array.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()
    
# Path to the image
image_path = './input/shoes.png'

# Preprocess the image
preprocessed_image = preprocess_image(image_path)
# Visualize the preprocessed image
visualize_image(preprocessed_image)

# Make a prediction
prediction = model.predict(preprocessed_image)
predicted_class = np.argmax(prediction, axis=1)
confidence_scores = prediction[0]

print("Prediction:", predicted_class)
print("Confidence Scores:", confidence_scores)