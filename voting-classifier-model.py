import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


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
    model2 = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
    model3 = MultinomialNB()
    model4 = RandomForestClassifier(n_estimators=200, criterion = "entropy")
    model5 = LogisticRegression(class_weight = "balanced", solver = "sag", multi_class="multinomial")
    model6 = BernoulliNB()
    model = VotingClassifier(estimators=[('xgb', model1), ('knn', model2), ('mnb', model3), ('rf', model4), ('lr', model5), ("bnb", model6)], voting='soft')
    model = OneVsRestClassifier(model)
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)

    return acc

X_train, X_test, Y_train, Y_test = process_data_for_model()
acc = train_and_test_models(X_train, X_test, Y_train, Y_test)
print(acc)
