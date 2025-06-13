# Hunter Downey
# 6 - 13 - 25
# model script

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import clean_and_engineer

# Suppress future downcasting warning
pd.set_option('future.no_silent_downcasting', True)

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

# Read data
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

# Clean and prepare
X = clean_and_engineer(train.drop(columns=["Transported"]))
y = train["Transported"].map({True: 1, False: 0})

X_test = clean_and_engineer(test, is_train=False)

# Drop PassengerId before prediction
if "PassengerId" in X_test.columns:
    passenger_ids = X_test["PassengerId"]
    X_test = X_test.drop(columns=["PassengerId"])
else:
    passenger_ids = test["PassengerId"]

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X, y)
preds = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": preds.astype(bool)
})
submission.to_csv("../submission.csv", index=False)
print("submission.csv generated.")

