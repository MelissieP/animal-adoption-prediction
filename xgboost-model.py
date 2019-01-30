import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_data_for_model():
    data = pd.read_csv("data.csv")
    data = pd.get_dummies(data, columns = ["Gender", "MaturitySize",
                                             "Vaccinated", "Dewormed", "Sterilized",
                                             "Health", 'Quantity','Color1', 'Color2', 'Color3', "FurLength", 'Breed1', 'Breed2'])
    y = data["AdoptionSpeed"]


    data = data.drop(
        columns=["AdoptionSpeed", "PetID", "Name", "State", "RescuerID", "Description", "BreedName_1", "BreedName_2",
                 "Fee", "Description_Character_Count", "Description_Word_Count", "Age"])

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)


    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    return X_train, X_test, Y_train, Y_test

def train_and_test_model(X_train, X_test, Y_train, Y_test):
    xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                    silent=True, nthread=1, num_class = 5,
                    max_depth = 20, gamma = 0.5, colsample_bytree = 1.0, min_child_weight = 5,
                    subsample = 0.8, n_estimators = 300)

    model = xgb.fit(X_train, Y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(Y_test, pred)

    return acc

X_train, X_test, Y_train, Y_test = process_data_for_model()
acc = train_and_test_model(X_train, X_test, Y_train, Y_test)
print(acc)
