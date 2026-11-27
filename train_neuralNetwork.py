import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split    
from sklearn.metrics import root_mean_squared_log_error 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.neural_network import MLPRegressor

# Load the dataset
train = pd.read_csv("train.csv")

# Take out the fetures that will not be at testing
y = np.log1p(train["hours_spent"])

X = train.drop(
    columns=["hours_spent", "log_hours", "hours_bucket", "project_id"]
)

# Find numerical and categorical features
categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Split into train ans validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Replace missing numerical values by the median
numerical_pipeline = Pipeline([
    (
        "imputer", 
        SimpleImputer(strategy="median")
    ),
    (
        "scaler", 
        StandardScaler()
    )
])

# Prepare Categorical values
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="constant", fill_value="Missing")
    ),
    (
        "convert_to_string",
        FunctionTransformer(lambda x: x.astype(str))
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Apply the pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# Create the neural network
nn = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.01,
    learning_rate_init=0.0005,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=40,
    random_state=42,
    verbose=True
)

# Make the model by combining preprocessing and neural network
model = Pipeline([
    ("preprocessing", preprocessor),
    ("neural_network", nn)
])

# Train the neural network
model.fit(X_train, y_train)

# Predict on the validation set
pred_log = model.predict(X_val)

# Convert predictions to hours
predictions = np.expm1(np.maximum(pred_log, 0))

# Evaluate the model
actual_hours = np.expm1(y_val)

score = root_mean_squared_log_error(
    actual_hours,
    predictions
)

print("\n" + "=" * 50)
print("NEURAL NETWORK RESULTS")
print("=" * 50)

print("Validation RMSLE:", score) # Validation RMSLE: 0.7254174344274544

# Compare indivisual predictions
results = pd.DataFrame({
    "Actual":actual_hours,
    "Predicted":predictions
})

results["Absolute_Error"] = abs(
    results["Actual"] - results["Predicted"]
)

results = results.sort_values(
    "Absolute_Error", 
    ascending=False
)

print("\nLargest prediction errors:")
print(results.head(20))