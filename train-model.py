import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from data_loader import load_data  # Import your data loader
from model import baseline_model  # Import your model

rows, cols = 28, 28 
train_data, test_data, train_labels, test_labels = load_data(rows, cols)


train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.2)

batch_size = 256
epochs = 5
input_shape = (rows, cols, 1)

model = baseline_model(input_shape)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y))

model.save('./image-class-model.h5')

predictions= model.predict(test_data)
