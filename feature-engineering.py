import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Load data
train = pd.read_csv("data/processed/train/train.csv", index_col=0)
test = pd.read_csv("data/processed/test/test.csv", index_col=0)


def create_character_quantile_columns(df):
    character_25 = df["Description_Character_Count"].quantile(q=0.25)
    character_5 = df["Description_Character_Count"].quantile(q=0.5)
    character_75 = df["Description_Character_Count"].quantile(q=0.75)

    df["Character_25"] = 0
    df.loc[df["Description_Character_Count"] <= character_25, "Character_25"] = 1

    df["Character_5"] = 0
    df.loc[ (df["Description_Character_Count"] > character_25) & (df["Description_Character_Count"] <= character_5), "Character_5"] = 1

    df["Character_75"] = 0
    df.loc[(df["Description_Character_Count"] > character_5) & (df["Description_Character_Count"] <= character_75), "Character_75"] = 1

    df["Character_100"] = 0
    df.loc[(df["Description_Character_Count"] > character_75), "Character_100"] = 1

    return df

def create_word_quantile_columns(df):
    word_25 = df["Description_Character_Count"].quantile(q=0.25)
    word_5 = df["Description_Character_Count"].quantile(q=0.5)
    word_75 = df["Description_Character_Count"].quantile(q=0.75)

    df["Character_25"] = 0
    df.loc[df["Description_Character_Count"] <= word_25, "Character_25"] = 1

    df["Character_5"] = 0
    df.loc[ (df["Description_Character_Count"] > word_25) & (df["Description_Character_Count"] <= word_5), "Character_5"] = 1

    df["Character_75"] = 0
    df.loc[(df["Description_Character_Count"] > word_5) & (df["Description_Character_Count"] <= word_75), "Character_75"] = 1

    df["Character_100"] = 0
    df.loc[(df["Description_Character_Count"] > word_75), "Character_100"] = 1
    return df