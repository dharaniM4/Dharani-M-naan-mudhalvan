import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

# Load new data
new_data = pd.read_csv("new_data.csv")  # Must match format

# Preprocess
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=columns, fill_value=0)

# Predict
predictions = model.predict(new_data)
print("Predicted prices:", predictions)
