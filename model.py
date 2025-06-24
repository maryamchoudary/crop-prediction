import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn

print("✅ scikit-learn version:", sklearn.__version__)

# Sample dataset for retraining
data = {
    "N": [90, 85, 60, 74],
    "P": [42, 58, 55, 35],
    "K": [43, 41, 44, 40],
    "temperature": [20.8, 21.7, 23.0, 23.4],
    "humidity": [82, 80, 82, 80],
    "ph": [6.5, 7.0, 7.8, 6.4],
    "rainfall": [202, 226, 263, 191],
    "label": ["rice", "rice", "maize", "maize"]
}

df = pd.DataFrame(data)

X = df.drop("label", axis=1)
y = df["label"]

# Train and save
model = DecisionTreeClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
