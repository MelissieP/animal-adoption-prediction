import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Flatten, Dense, concatenate, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import utils
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read in the data and assign the target variable to y
data = pd.read_csv("data/processed/embedding_data.csv")
y = data["AdoptionSpeed"]

# Create train test split for measuring accuracy
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.3, random_state=1)

# Encode the target variable
encoder = LabelEncoder()
encoder.fit(y)
y_train = encoder.transform(Y_train)
y_test = encoder.transform(Y_test)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# Get the columns that won't be embedded. These will be passed through its own layer
data_train = X_train[["Type_1", "Type_2", "Age", "Sentiment_Score", "Sentiment_Magnitude", "Photo_Score", "MixedBreed", "Unnamed"]]
X_test = X_test[["Type_1", "Type_2", "Age", "Sentiment_Score", "Sentiment_Magnitude", "Photo_Score", "MixedBreed", "Unnamed"]]

data_input = Input(shape=(8,), name = "data_train")
data_layer = Dense(16,activation = "relu", input_shape=(8,))(data_input)

# Set embedding size
embedding_size = 3

# Create input layers of the features that will be embedded
colour1_input = Input(shape=(1,), name='colour1')
colour2_input = Input(shape=(1,), name='colour2')
colour3_input = Input(shape=(1,), name='colour3')

gender_input = Input(shape=(1,), name='gender')
vaccinated_input = Input(shape=(1,), name='vaccinated')
dewormed_input = Input(shape=(1,), name='dewormed')
sterilised_input = Input(shape=(1,), name='sterilised')
health_input = Input(shape=(1,), name='health_1')
size_input = Input(shape=(1,), name = "size")
fur_length_input = Input(shape=(1,), name = "fur_length")

breed1_input = Input(shape=(1,), name='breed1')
breed2_input = Input(shape=(1,), name='breed2')

# Create embedding layer
colour1_embedded = Embedding(X_train["Color1"].max()+1, embedding_size,
                                       name='colour1_embedding')(colour1_input)
colour2_embedded = Embedding(X_train["Color2"].max()+1, embedding_size,
                                       name='colour2_embedding')(colour2_input)
colour3_embedded = Embedding(X_train["Color3"].max()+1, embedding_size,
                                        input_length=1, name='colour3_embedding')(colour3_input)
breed1_embedded = Embedding(X_train["Breed1"].max()+1, embedding_size,
                                       input_length=1, name='breed1_embedding')(breed1_input)
breed2_embedded = Embedding(X_train["Breed2"].max()+1, embedding_size,
                                       input_length=1, name='breed2_embedding')(breed2_input)
gender_embedded = Embedding(X_train["Gender"].max()+1, embedding_size,
                                       input_length=1, name='gender_embedding')(gender_input)
vaccinate_embedded = Embedding(X_train["Vaccinated"].max()+1, embedding_size,
                                       input_length=1, name='vaccinate_embedding')(vaccinated_input)
deworm_embedded = Embedding(X_train["Dewormed"].max()+1, embedding_size,
                                       input_length=1, name='deworm_embedding')(dewormed_input)
sterilised_embedded = Embedding(X_train["Sterilized"].max()+1, embedding_size,
                                        input_length=1, name='sterilised_embedding')(sterilised_input)
health_embedded = Embedding(X_train["Health"].max()+1, embedding_size,
                                       input_length=1, name='health_embedding')(health_input)
size_embedded = Embedding(X_train["MaturitySize"].max()+1, embedding_size,
                                       input_length=1, name='size_embedding')(health_input)
fur_embedded = Embedding(X_train["FurLength"].max()+1, embedding_size,
                                       input_length=1, name='fur_embedding')(fur_length_input)

# Concatenate the embeddings and remove the extra dimension with Flatte()
concatenated = concatenate([colour1_embedded, colour2_embedded, colour3_embedded,
                            breed1_embedded, breed2_embedded, gender_embedded,
                              vaccinate_embedded, deworm_embedded, sterilised_embedded,
                              health_embedded, size_embedded, fur_embedded])
x = Flatten()(concatenated)
#Concatenate the embedded layers with the dense layer of the remaining data
x = concatenate([x, data_layer])
x = Dropout(0.2)(Dense(40, activation="relu")(x))
x = Dropout(0.2)(Dense(20, activation="relu")(x))
x = Dropout(0.1)(Dense(10, activation="relu")(x))
# Predicting the 5 classes
out = Dense(5, activation="softmax", name="prediction")(x)

# What input the model can expect
model = Model(
    inputs = [colour1_input, colour2_input, colour3_input,
              breed1_input, breed2_input, gender_input, vaccinated_input, dewormed_input, sterilised_input, health_input,
              size_input, fur_length_input, data_input],
    outputs=out)

model.summary(line_length=88)

model.compile(loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
              optimizer= tf.train.AdamOptimizer(0.005))

# Fit the model
history = model.fit([X_train["Color1"], X_train["Color2"], X_train["Color3"], X_train["Breed1"], X_train["Breed2"],
                     X_train["Gender"], X_train["Vaccinated"], X_train["Dewormed"],
                     X_train["Sterilized"], X_train["Health"], X_train["MaturitySize"], X_train["FurLength"], data_train],
                    y_train, batch_size=10,
                    epochs=280,
                    verbose=1,
                    validation_split=.5)

# Evaluating the model
score = model.evaluate([X_test["Color1"], X_test["Color2"], X_test["Color3"], X_test["Breed1"], X_test["Breed2"],
                        X_test["Gender"], X_test["Vaccinated"], X_test["Dewormed"],
                        X_test["Sterilized"], X_test["Health"], X_test["MaturitySize"], X_test["FurLength"], X_test], y_test,
                        batch_size=500, verbose=1)
print('Test accuracy:', score[1] * 100)
