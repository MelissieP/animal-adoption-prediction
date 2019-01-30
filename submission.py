import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras import utils
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def process_data_for_model():
    data = pd.read_csv("data.csv")
    data = pd.get_dummies(data, columns = ["Gender", "MaturitySize",
                                             "Vaccinated", "Dewormed", "Sterilized",
                                             "Health", 'Quantity'])
    y = data["AdoptionSpeed"]


    data = data.drop(
        columns=["AdoptionSpeed", "PetID", "Name", "State", "RescuerID", "Description", "BreedName_1", "BreedName_2",
                 "Fee", "Description_Character_Count", "Description_Word_Count", "Age",  'Color1', 'Color2', 'Color3', "FurLength", 'Breed1', 'Breed2'])

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)


    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.3)

    #  Separate target variable from training set, and remove unnecessary columns
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    y_train = encoder.transform(Y_train)
    y_test = encoder.transform(Y_test)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test, num_classes

def build_model(X_train, X_test, y_train, y_test, num_classes):
    batch_size = 258
    epochs = 500

    # Build the model
    model = Sequential()
    model.add(Dense(114, input_shape=(57,)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.1))

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.1))

    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.1))

    model.add(Dense(25))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2)

    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1] * 100)

    return score[1] * 100

X_train, X_test, y_train, y_test, num_classes = process_data_for_model()
test_accuracy = build_model(X_train, X_test, y_train, y_test, num_classes)