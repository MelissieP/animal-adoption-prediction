import warnings
warnings.filterwarnings('ignore')
import pandas as pd

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

def save_processed_data(train, test, path):

    train = train.drop(columns=["Name", "State", "RescuerID", "Description", "PetID", "BreedName_1", "BreedName_2",
                                "Fee", "Description_Character_Count", "Description_Word_Count", "Age"])

    test = test.drop(columns=["Name", "State", "RescuerID", "Description", "PetID", "BreedName_1", "BreedName_2",
                              "Fee", "Description_Character_Count", "Description_Word_Count", "Age"])

    train.to_csv(path + "train/train.csv", index = False)
    test.to_csv(path + "test/test.csv", index = False)
    return


def engineer_features():
    train = pd.read_csv("data/pre-processed/train/train.csv", index_col=0)
    test = pd.read_csv("data/pre-processed/test/test.csv", index_col=0)

    train = create_character_quantile_columns(train)
    test = create_character_quantile_columns(test)

    train = create_word_quantile_columns(train)
    test = create_word_quantile_columns(test)

    train = bucket_ages(train)
    test = bucket_ages(test)

    train = bucket_fees(train)
    test = bucket_fees(test)


    print("\nSaving data")
    save_processed_data(train, test, "data/processed/")
    print("\nData saved")
    return

engineer_features()