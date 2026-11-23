# Seperate the dataset into 4000 training and 500 testing radomly
import pandas as pd

df=pd.read_csv("officialTrain.csv")

testing = df.sample(n=500, random_state=42)

training = df.drop(testing.index)

testing.to_csv("testing.csv")
training.to_csv("train.csv")