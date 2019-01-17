import pandas as pd

train = pd.read_csv("data/train/train.csv")

train.drop(labels = "Description", axis = 1, inplace=True)