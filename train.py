# Predicts how many hours a project took using CatBoost

# Import tools:
import pandas as pd # Manipulate & Read data in tables
import numpy as np  # Used for arrays, logaritms, and expoenentials
import catboost     # Machine Learning alghoritm that builds decison trees

from catboost import CatBoostRegressor                  # Machine learning model to predict numerical values
from sklearn.model_selection import train_test_split    # To divide the dataset into train and test
from sklearn.metrics import root_mean_squared_log_error # To calculate RMSLE(Root Mean Squared Logarithmic Error)

# Load the dataset
train = pd.read_csv("train.csv")

# Transform the 'hours spent' using: y = ln(1 + hours_spent). To avoid outliers affecting training
y = np.log1p(train["hours_spent"])

# Drop uneccesary values: hours_spent & log_hours & hours_bucket(the answer), project_id(not relevant to prediction)
X = train.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])

# Idetify categorical features: programming language, project name... etc
categorical_features = X.select_dtypes(include=["str", "object"]).columns.tolist()

# Handle missing categorical values
for col in categorical_features:
    X[col] = X[col].fillna("Missing").astype(str)

# Split into training and testing data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model: 
model = CatBoostRegressor(
    iterations=1000,    # How many iterations to do
    depth=10,           # How deep each tree can get
    learning_rate=0.05, # How much weights are adjusted
    verbose=50          # Print progress every 50 iterations
)

# Train the model: 
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features, # Pass in categorical categories for catboost
    eval_set=(X_val, y_val),
    early_stopping_rounds=100          # Stop after 100 iterations if performance does not improve
)

# Predict hours but in logged form: y = ln(1 + hours_spent)
pred_log = model.predict(X_val)

# Reverse Logaritmic transformation to get hours: hours = exp(y) - 1
predictions = np.expm1(np.maximum(pred_log, 0))

# Measure model accuracy: Convert back y_val back to actual hours and calculate RMSLE
score = root_mean_squared_log_error(np.expm1(y_val), predictions)
print("Validation RMSLE:", score)

# Reccord correct hours and predicted
results = pd.DataFrame({
    "Actual": np.expm1(y_val),
    "Predicted": predictions
})
print(results.head(20))