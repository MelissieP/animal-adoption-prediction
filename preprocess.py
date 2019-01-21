import pandas as pd
import numpy as np

def read_in_data(path):
    train = pd.read_csv(path + "train/train.csv")

    test = pd.read_csv(path + "test/test.csv")
    submission = pd.read_csv(path + "/test/sample_submission.csv")
    test = pd.merge(test, submission, on="PetID")

    breeds = pd.read_csv(path + "breed_labels.csv")
    colours = pd.read_csv(path + "color_labels.csv")
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
    train = train.dropna()
    test = test.dropna()
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

def save_processed_data(train, test, path):
    train.to_csv(path + "train/train.csv", index = False)
    test.to_csv(path + "test/test.csv", index = False)
    return

def process_data():
    print("\nReading in the data")
    train, test, breeds, colours = read_in_data("data/")

    print(train["AdoptionSpeed"].count())
    print("\nStandardise pets with no names")
    train, test = standardise_unnamed_pets(train, test)

    print(train["AdoptionSpeed"].count())
    train, test = create_unnamed_feature(train, test)
    print("\nDropping NA values")

    print(train["AdoptionSpeed"].count())
    train, test = drop_na_values(train, test)
    print("\nCreating mixed breed feature")

    print(train["AdoptionSpeed"].count())
    train, test = create_mixed_breed_feature(train, test)
    breed_dict = breed_df_to_dict(breeds)
    train, test = map_dictionary_to_breed_names(breed_dict, train, test)

    print(train["AdoptionSpeed"].count())
    print("\nMapping breed dictionary to data")
    train, test = create_description_length_feature(train, test)
    print("\nCreated description length feature")

    print(train["AdoptionSpeed"].count())
    print("\nSaving data")
    save_processed_data(train, test, "data/processed/")
    print("\nData saved")
    return

process_data()
