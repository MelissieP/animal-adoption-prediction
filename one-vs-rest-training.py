import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score

def process_data_for_model():
    data = pd.read_csv("data/processed/data.csv")
    data = data.loc[data["Type"] == 2]

    y = data["AdoptionSpeed"]
    data = data.drop(
        columns=["AdoptionSpeed"])


    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    return X_train, X_test, Y_train, Y_test

def train_and_test_models(X_train, X_test, Y_train, Y_test):
    model1 = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                        silent=True, nthread=1, num_class=5,
                        max_depth=15, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                        subsample=0.8, n_estimators=200)
    model = OneVsRestClassifier(model1)
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    pred = model.predict(X_test)

    return acc, pred

X_train, X_test, Y_train, Y_test = process_data_for_model()
pred, acc = train_and_test_models(X_train, X_test, Y_train, Y_test)

print(acc)

Y_test = Y_test.values

kappa = cohen_kappa_score(Y_test, pred)
