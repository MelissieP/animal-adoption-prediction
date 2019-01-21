from numpy import loadtxt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
train = pd.read_csv("data/processed/train/train.csv", index_col=0)
test = pd.read_csv("data/processed/test/test.csv", index_col=0)

#  Separate target variable from training set
Y = train["AdoptionSpeed"]
X = train.drop(columns = ["AdoptionSpeed", "Name", "State", "RescuerID", "Description", "PetID", "BreedName_1", "BreedName_2"])