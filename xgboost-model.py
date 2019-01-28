import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def read_in_data():
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    breeds = pd.read_csv("input/breed_labels.csv")
    colours = pd.read_csv("input/color_labels.csv")
    return train, test, breeds, colours

def standardise_unnamed_pets(train, test):
    train["Name"] = train["Name"].fillna("Unnamed")
    train["Name"] = np.where(train["Name"] == "No Name", "Unnamed", train["Name"])
    train["Name"] = np.where(train["Name"] == "None", "Unnamed", train["Name"])
    train["Name"] = np.where(train["Name"] == "No Name Yet", "Unnamed", train["Name"])

    test["Name"] = test["Name"].fillna("Unnamed")
    test["Name"] = np.where(test["Name"] == "No Name", "Unnamed", test["Name"])
    test["Name"] = np.where(test["Name"] == "None", "Unnamed", test["Name"])
    test["Name"] = np.where(test["Name"] == "No Name Yet", "Unnamed", test["Name"])
    return train, test

def create_unnamed_feature(train, test):
    train["Unnamed"] = 0
    train.loc[train["Name"] == "Unnamed", "Unnamed"] = 1

    test["Unnamed"] = 0
    test.loc[test["Name"] == "Unnamed", "Unnamed"] = 1
    return train, test

def drop_na_values(train, test):
    train = train.fillna(0)
    test = test.fillna(0)
    return train, test

def create_mixed_breed_feature(train, test):
    train["MixedBreed"] = [1 if x > 0 else 0 for x in train["Breed2"]]
    test["MixedBreed"] = [1 if x > 0 else 0 for x in test["Breed2"]]
    return train, test

def breed_df_to_dict(breeds):
    breed_cols = breeds[["BreedID", "BreedName"]]
    breed_cols.set_index("BreedID", drop=True, inplace=True)
    breed_dict = breed_cols.to_dict()["BreedName"]
    return breed_dict

def map_dictionary_to_breed_names(breed_dict, train, test):
    train["BreedName_1"] = train["Breed1"].map(breed_dict).fillna("Unknown")
    train["BreedName_2"] = train["Breed2"].map(breed_dict).fillna("None")

    test["BreedName_1"] = test["Breed1"].map(breed_dict).fillna("Unknown")
    test["BreedName_2"] = test["Breed2"].map(breed_dict).fillna("None")

    # Make all the cases pure bred if breed 1 is the same as breed 2
    train.loc[train["BreedName_1"] == train["BreedName_2"], "MixedBreed"] = 0
    test.loc[test["BreedName_1"] == test["BreedName_2"], "MixedBreed"] = 0

    # But for the following cases, they are mixed:
    train.loc[train["BreedName_1"] == "Mixed Breed", "MixedBreed"] = 1
    train.loc[train["BreedName_1"] == "Domestic Short Hair", "MixedBreed"] = 1
    train.loc[train["BreedName_1"] == "Domestic Medium Hair", "MixedBreed"] = 1
    train.loc[train["BreedName_1"] == "Domestic Long Hair", "MixedBreed"] = 1

    test.loc[test["BreedName_1"] == "Mixed Breed", "MixedBreed"] = 1
    test.loc[test["BreedName_1"] == "Domestic Short Hair", "MixedBreed"] = 1
    test.loc[test["BreedName_1"] == "Domestic Medium Hair", "MixedBreed"] = 1
    test.loc[test["BreedName_1"] == "Domestic Long Hair", "MixedBreed"] = 1

    return train, test

def create_description_length_feature(train, test):
    train["Description"] = train["Description"].fillna("")
    train["Description_Character_Count"] = train["Description"].apply(lambda x: len(x))
    train["Description_Word_Count"] = train["Description"].apply(lambda x: len(x.split()))

    test["Description"] = test["Description"].fillna("")
    test["Description_Character_Count"] = test["Description"].apply(lambda x: len(x))
    test["Description_Word_Count"] = test["Description"].apply(lambda x: len(x.split()))
    return train, test

train, test, breeds, colours = read_in_data()
train, test = standardise_unnamed_pets(train, test)
train, test = create_unnamed_feature(train, test)
train, test = create_mixed_breed_feature(train, test)
breed_dict = breed_df_to_dict(breeds)
train, test = map_dictionary_to_breed_names(breed_dict, train, test)
train, test = create_description_length_feature(train, test)

train, test = drop_na_values(train, test)

def create_character_quantile_columns(df):
    character_25 = df["Description_Character_Count"].quantile(q=0.25)
    character_5 = df["Description_Character_Count"].quantile(q=0.5)
    character_75 = df["Description_Character_Count"].quantile(q=0.75)

    df["Character_25"] = 0
    df.loc[df["Description_Character_Count"] <= character_25, "Character_25"] = 1

    df["Character_5"] = 0
    df.loc[(df["Description_Character_Count"] > character_25) & (
            df["Description_Character_Count"] <= character_5), "Character_5"] = 1

    df["Character_75"] = 0
    df.loc[(df["Description_Character_Count"] > character_5) & (
            df["Description_Character_Count"] <= character_75), "Character_75"] = 1

    df["Character_100"] = 0
    df.loc[(df["Description_Character_Count"] > character_75), "Character_100"] = 1

    return df

def create_word_quantile_columns(df):
    word_25 = df["Description_Word_Count"].quantile(q=0.25)
    word_5 = df["Description_Word_Count"].quantile(q=0.5)
    word_75 = df["Description_Word_Count"].quantile(q=0.75)

    df["Word_25"] = 0
    df.loc[df["Description_Word_Count"] <= word_25, "Word_25"] = 1

    df["Word_5"] = 0
    df.loc[(df["Description_Word_Count"] > word_25) & (
            df["Description_Word_Count"] <= word_5), "Word_5"] = 1

    df["Word_75"] = 0
    df.loc[(df["Description_Word_Count"] > word_5) & (
            df["Description_Word_Count"] <= word_75), "Word_75"] = 1

    df["Word_100"] = 0
    df.loc[(df["Description_Word_Count"] > word_75), "Word_100"] = 1
    return df

def bucket_ages(df):
    # Drop Ages thats = 0
    df = df[df["Age"] > 0]
    df["Puppy"] = 0
    df["Adult"] = 0
    df["Senior"] = 0
    df.loc[(df["Age"] >= 1) & (df["Age"] < 12), "Puppy"] = 1
    df.loc[(df["Age"] >= 12) & (df["Age"] < 96), "Adult"] = 1
    df.loc[(df["Age"] >= 96, "Senior")] = 1
    return df

def bucket_fees(df):
    fee_25 = df["Fee"].quantile(q=0.25)
    fee_5 = df["Fee"].quantile(q=0.5)
    fee_75 = df["Fee"].quantile(q=0.75)
    df["No_Fee"] = 0
    df["Fee_25"] = 0
    df["Fee_5"] = 0
    df["Fee_75"] = 0
    df["Fee_100"] = 0
    df.loc[df["Fee"] == 0, "No_Fee"] = 1
    df.loc[(df["Fee"] > 0) & (
            df["Fee"] <= fee_25), "Fee_25"] = 1
    df.loc[(df["Fee"] > fee_25) & (
            df["Fee"] <= fee_5), "Fee_5"] = 1
    df.loc[(df["Fee"] > fee_5) & (
            df["Fee"] <= fee_75), "Fee_75"] = 1
    df.loc[(df["Fee"] > fee_75), "Fee_100"] = 1
    return df

test = create_character_quantile_columns(test)
train = create_word_quantile_columns(train)
test = create_word_quantile_columns(test)
train = bucket_ages(train)
test = bucket_ages(test)
train = bucket_fees(train)
test = bucket_fees(test)


y = train["AdoptionSpeed"]


train = train.drop(
    columns=["AdoptionSpeed", "PetID", "Name", "State", "RescuerID", "Description", "BreedName_1", "BreedName_2",
            "Fee", "Description_Character_Count", "Description_Word_Count", "Age"])

data = pd.get_dummies(train, columns = ["Type", "Gender", "MaturitySize", "FurLength",
                                         "Vaccinated", "Dewormed", "Sterilized",
                                         "Health", 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Quantity'])

X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.33, random_state=1)

model = XGBClassifier()
model.fit(X_train, Y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))