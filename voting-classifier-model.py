import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score


def process_data_for_model():
    data = pd.read_csv("data/processed/data2.csv")
    y = data["AdoptionSpeed"]
    data = data.drop(
        columns=["AdoptionSpeed"])


    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.3, random_state=1)

    return X_train, X_test, Y_train, Y_test

def train_and_test_models(X_train, X_test, Y_train, Y_test):
    model1 = XGBClassifier(learning_rate=0.021,  objective='multi:softmax',
                        silent=True, nthread=1, num_class=5,
                        max_depth=10, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                        subsample=0.8, n_estimators=100)

    model7 = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                        silent=True, nthread=1, num_class=5,
                        max_depth=15, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                        subsample=0.8, n_estimators=200)

    model8 = XGBClassifier(learning_rate=0.05, objective='multi:softmax',
                        silent=True, nthread=1, num_class=5,
                        max_depth=30, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                        subsample=0.8, n_estimators=300)
    model9= XGBClassifier(learning_rate=0.1, objective='multi:softmax',
                        silent=True, nthread=1, num_class=5,
                        max_depth=50, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                        subsample=0.8, n_estimators=500)
    model2 = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
    model3 = MultinomialNB()
    model4 = RandomForestClassifier(n_estimators=200, criterion = "entropy")
    model5 = LogisticRegression(class_weight = "balanced", solver = "sag", multi_class="multinomial")
    model6 = BernoulliNB()
    # model7 = NuSVC(gamma = "scale")
    # model8 = SVC()
   # model10 = NearestCentroid()
    model11 = BaggingClassifier(n_estimators = 30, max_samples = 5, max_features = 5, bootstrap_features=True)
    model = VotingClassifier(estimators=[('xgb', model1), ("bag", model11), ('xgb2', model7),
                                         ('xgb3', model8), ('xgb4', model9),
                                         ('knn', model2), ('mnb', model3), ('rf', model4), ('lr', model5),
                                         ("bnb", model6)], voting='soft') # ('nusvc', model7), ('svc', model8),  ("nc", model10),, ('knn', model2), ('mnb', model3), ('rf', model4), ('lr', model5), ("bnb", model6),
    model = OneVsRestClassifier(model)
    model = model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    pred = model.predict(X_test)

    return acc, pred

X_train, X_test, Y_train, Y_test = process_data_for_model()
acc, pred = train_and_test_models(X_train, X_test, Y_train, Y_test)
print(acc)

Y_test = Y_test.values

kappa = cohen_kappa_score(Y_test, pred)
print(kappa)
