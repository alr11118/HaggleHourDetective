# Use CatBoost: https://www.geeksforgeeks.org/machine-learning/catboost-ml/
import pandas as pd
import numpy as np

import catboost as CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error

train = pd.read_csv("train.csv")

y = np.log1p(train["hours_spent"])
X = train.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])

features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=1000, 
                          depth=6, 
                          learning_rate=0.05, 
                          verbose=100)

model.fit(X_train, y_train, catFeatures=features, eval_set=(X_val, y_val), early_stopping_rounds=100)

pred_log = model.predict(X_val)
predictions = np.expm1(np.maximum(pred_log, 0))

score = root_mean_squared_log_error(np.expm1(y_val), predictions)