import pandas as pd
import numpy as np

def read_in_data(path):
    train = pd.read_csv(path + "train/train.csv")
    test = pd.read_csv(path + "test/test.csv")
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
    train.dropna(axis=0, how="any", inplace=True)
    test.dropna(axis=0, how="any", inplace=True)
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

    return test, train

def save_processed_data(train, test, path):
    train.to_csv(path + "train/train.csv", index = False)
    test.to_csv(path + "test/test.csv", index = False)
    return

def process_data():
    print("\nReading in the data ...")
    train, test, breeds, colours = read_in_data("data/")
    print("\nData read in")
    train, test = standardise_unnamed_pets(train, test)
    print("\nPets with no names standardised")
    train, test = create_unnamed_feature(train, test)
    train, test = drop_na_values(train, test)
    print("\nDrop NA values")
    train, test = create_mixed_breed_feature(train, test)
    breed_dict = breed_df_to_dict(breeds)
    test, train = map_dictionary_to_breed_names(breed_dict, train, test)
    print("\nMapping breed dictionary to data ...")
    print("\nSaving data...")
    save_processed_data(train, test, "data/processed/")
    print("\nData saved")
    return

process_data()
