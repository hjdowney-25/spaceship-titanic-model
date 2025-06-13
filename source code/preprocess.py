# Hunter Downey
# 6 - 13 - 25
# Preprocess script

import pandas as pd

def clean_and_engineer(df, is_train=True):
    df = df.copy()
    if "Cabin" in df.columns: df.drop(columns=["Cabin", "Name"], inplace=True)
    
    # Fill missing values
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(df[col].median())

    # Encode binary values for categorical data
    df['CryoSleep'] = df['CryoSleep'].map({'True': 1, 'False': 0})
    df['VIP'] = df['VIP'].map({'True': 1, 'False': 0})

    # One-hot encode
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], drop_first=True)

    # Feature: total spend
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

    # Optional: drop PassengerId if not needed
    if "PassengerId" in df.columns and is_train:
        df.drop(columns=["PassengerId"], inplace=True)
    return df
