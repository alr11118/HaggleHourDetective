import numpy as np
import pandas as pd
from catboost import CatBoostRegressor 

# Load data
test = pd.read_csv("test.csv")

# Reccord project Ids
if "project_id" in test.columns:
    project_ids = test["project_id"].copy()
else:
    project_ids = pd.Series(range(len(test)), name="project_id")

# Remove bad columns
X = test.drop(
    columns=["project_id", "hours_spent", "log_hours", "hours_bucket"],
    errors="ignore"
)

# Load model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Match test feaures to training features
model_features = model.feature_names_
X = X.reindex(columns=model_features)

# Handle missing values
categorical_features = model.get_cat_feature_indices()
for i in categorical_features:
    col = model_features[i]
    X[col] = X[col].fillna("Missing").astype(str)

# Make predictions
pred_log = model.predict(X)

# Convert predictions to hours
predictions = np.expm1(np.maximum(pred_log, 0))
# Limit predtions at testing max
predictions = np.clip(predictions, 0, 140.7)

# Reccord predictions
output = pd.DataFrame({"project_id": project_ids, "hours_spent": predictions})
output.to_csv("predictions.csv", index=False)

# Print Resaukts
print("Predictions saved to predictions.csv")
print("\nFirst 20 preditions")
print(output.head(20))
print("\nNumber of predictions:", len(predictions))
