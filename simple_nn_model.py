import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

# Load data
train = pd.read_csv("data/processed/train/train.csv")
test = pd.read_csv("data/processed/test/test.csv")

#  Separate target variable from training set, and remove unnecessary columns
Y_train = train["AdoptionSpeed"]
X_train = train.drop(
    columns=["AdoptionSpeed"])

Y_test = test["AdoptionSpeed"].astype(int)
X_test = test.drop(
    columns=["AdoptionSpeed"])

encoder = LabelEncoder()
encoder.fit(Y_train)
y_train = encoder.transform(Y_train)
y_test = encoder.transform(Y_test)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


batch_size = 256
epochs = 400
learning_rate = 0.0001

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(43,)))
model.add(Activation("relu"))
model.add(Dense(300))
model.add(Activation("relu"))
model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(20))
model.add(Activation("relu"))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = "tf_log_dir",
        histogram_freq = 0,
        embeddings_freq = 0,
    )
]

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split = 0.25,
                    callbacks = callbacks)

score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1] * 100)