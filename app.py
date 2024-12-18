import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
file_path = "Housing.csv"
data = pd.read_csv(file_path)

# Define independent (X) and dependent (Y) variables
X = data.drop("price", axis=1)  # Drop the target variable
y = data["price"]  # Target variable

# Identify continuous and categorical variables
continuous_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

# Preprocess the data
# Apply standard scaling to continuous variables and one-hot encoding to categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ]
)

# Preprocess the independent variables
X_transformed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Display coefficients
feature_names = (
    continuous_features
    + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
)
coefficients = model.coef_

print("\nFeature Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
