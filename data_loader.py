import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow import keras

from sklearn.model_selection import train_test_split 

import tensorflow as tf

# Import necessary modules from TensorFlow Keras
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.layers import Flatten, BatchNormalization # type: ignore


def load_data(rows, cols):
    # Load the training and testing datasets from CSV files
    train_df = pd.read_csv('./dataset/fashion-mnist_train.csv')
    test_df = pd.read_csv('./dataset/fashion-mnist_test.csv')

    # Print the first few rows of the training dataset to verify loading
    print(train_df.head())

    # Convert the training and testing data to numpy arrays, excluding the first column (labels)
    train_data = np.array(train_df.iloc[:, 1:])
    test_data = np.array(test_df.iloc[:, 1:])

    # Convert the labels to one-hot encoded format
    train_labels = to_categorical(train_df.iloc[:, 0])
    test_labels = to_categorical(test_df.iloc[:, 0])

    # Reshape the data to fit the model input requirements (number of samples, rows, cols, channels)
    train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)
    test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)

    # Convert the data type to float32 for compatibility with TensorFlow
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    train_data /= 255.0
    test_data /= 255.0
    
    return train_data, test_data, train_labels, test_labels