from numpy import loadtxt
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.metrics import classification_report

# Load data
train = pd.read_csv("data/processed/train/train.csv", index_col=0)
test = pd.read_csv("data/processed/test/test.csv", index_col=0)

#  Separate target variable from training set, and remove unnecessary columns
Y_train = train["AdoptionSpeed"]
X_train = train.drop(columns = ["AdoptionSpeed", "Name", "State", "RescuerID", "Description", "PetID", "BreedName_1", "BreedName_2"])

Y_test = test["AdoptionSpeed"]
X_test = test.drop(columns = ["AdoptionSpeed", "Name", "State", "RescuerID", "Description", "PetID", "BreedName_1", "BreedName_2"])

# Let's train the initial model
model = xgb.XGBClassifier(n_estimators=100, max_depth = 15, learning_rate=0.1, subsample = 0.5)
train_model = model.fit(X_train, Y_train)

predictions = train_model.predict(X_test)


print('Model Report %r' % (classification_report(Y_test, predictions)))