import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
train = pd.read_csv("data/processed/train/train.csv")
test = pd.read_csv("data/processed/test/test.csv")

#  Separate target variable from training set, and remove unnecessary columns
Y_train = train["AdoptionSpeed"]
X_train = train.drop(
    columns=["AdoptionSpeed"])

Y_test = test["AdoptionSpeed"]
X_test = test.drop(
    columns=["AdoptionSpeed"])

# Model training
model = RandomForestClassifier(n_estimators = 20)


model = model.fit(X_train, Y_train)

# make predictions for test data
y_pred = model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
print(acc)

