import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.linear_model import LogisticRegression

def process_data_for_model():
    data = pd.read_csv("data/processed/data.csv")

    y = data["AdoptionSpeed"]
    data = data.drop(
        columns=["AdoptionSpeed"])


    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = process_data_for_model()

lr = LogisticRegression()
names = ['Random Forest', 'Extra Trees', 'KNeighbors', 'SVC', 'Ridge Classifier']


def zip_stacked_classifiers(*args):
    to_zip = []
    for arg in args:
        combined_items = sum([map(list, combinations(arg, i)) for i in range(len(arg) + 1)], [])
        combined_items = filter(lambda x: len(x) > 0, combined_items)
        to_zip.append(combined_items)

    return zip(to_zip[0], to_zip[1])


stacked_clf_list = zip_stacked_classifiers(clf_array, names)
best_combination = [0.00, ""]
for clf in stacked_clf_list:

    ensemble = SuperLearner(scorer=accuracy_score,
                            random_state=seed,
                            folds=10)
    ensemble.add(clf[0])
    ensemble.add_meta(lr)
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    accuracy = accuracy_score(preds, y_test)

    if accuracy > best_combination[0]:
        best_combination[0] = accuracy
        best_combination[1] = clf[1]

    print("Accuracy score: {:.3f} {}").format(accuracy, clf[1])
print("\nBest stacking model is {} with accuracy of: {:.3f}").format(best_combination[1], best_combination[0])