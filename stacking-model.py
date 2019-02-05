import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score

def process_data_for_model():
    data = pd.read_csv("data/processed/data.csv")
    y = data["AdoptionSpeed"]
    data = data.drop(
        columns=["AdoptionSpeed"])

    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2, random_state=1)
    return X_train, X_test, Y_train, Y_test


def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        test_pred=np.append(test_pred,model.predict(test))

    return test_pred.reshape(-1,1),train_pred

X_train, X_test, Y_train, Y_test = process_data_for_model()

model1 = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                       silent=True, nthread=1, num_class=5,
                       max_depth=15, gamma=0.5, colsample_bytree=1.0, min_child_weight=5,
                       subsample=0.8, n_estimators=200)
model1 = OneVsRestClassifier(model1)

model2 = KNeighborsClassifier()

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=X_train,test=X_test,y=Y_train)
train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)


test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=X_train,test=X_test,y=Y_train)
train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(multi_class="multinomial")
model.fit(df,Y_train)
model.score(df_test, Y_test)

pred = model.predict(df_test)
Y_test = Y_test.values

kappa = cohen_kappa_score(Y_test, pred)
print(kappa)
