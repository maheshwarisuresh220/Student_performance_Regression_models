import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# ======================
# Load and Prepare Dataset
# ======================
dataset = pd.read_csv("StudentPerformanceFactors.csv")

# Standardize column names to avoid errors with inconsistent naming
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(" ", "_")

print("=== Initial Dataset Preview ===")
print(dataset.head(), "\n")
rows, cols = dataset.shape
print(f"Dataset shape: {rows} rows, {cols} columns")
print("Columns:", list(dataset.columns), "\n")

# Drop irrelevant columns if they exist
to_drop = [c for c in ["family_income", "distance_from_home", "gender", "physical_activity"] if c in dataset.columns]
if to_drop:
    print(f"Dropping columns: {to_drop}")
    dataset = dataset.drop(columns=to_drop)

# Remove rows with missing values (simple approach)
dataset = dataset.dropna()

# ======================
# Feature and Target Selection
# ======================
feature_col = "hours_studied"
target_col = "exam_score"

if feature_col not in dataset.columns or target_col not in dataset.columns:
    raise KeyError(f"Required columns '{feature_col}' or '{target_col}' not found. Available: {list(dataset.columns)}")

X = dataset[[feature_col]]  # Independent variable as 2D array
y = dataset[target_col]     # Dependent variable

# ======================
# Basic Data Visualization
# ======================
# Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(dataset[[feature_col, target_col]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Study Hours vs Exam Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Pairplot to visualize relationship
sns.pairplot(dataset[[feature_col, target_col]])
plt.suptitle("Pairplot: Hours Studied vs Exam Score", fontsize=14, fontweight="bold")
plt.subplots_adjust(top=0.9)
plt.show()

# Scatter plot with linear regression trend line
plt.figure()
sns.scatterplot(x=feature_col, y=target_col, data=dataset, alpha=0.7, label="Students")
sns.regplot(x=feature_col, y=target_col, data=dataset, scatter=False, color="red", label="Linear Trend")
plt.title("Study Hours vs Exam Score", fontsize=14, fontweight="bold")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.show()

# ======================
# Linear Regression Model
# ======================
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

# Train model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred = linear_model.predict(X_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression -> RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Cross-validated predictions
cv_pred = cross_val_predict(linear_model, X, y, cv=5)
cv_rmse = np.sqrt(mean_squared_error(y, cv_pred))
print(f"5-fold Cross-Validation RMSE: {cv_rmse:.3f}\n")

# Predicted vs True Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="w", label="Predictions")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
plt.xlabel("True Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Model Predictions vs Actual Results", fontsize=14, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()

# ======================
# Polynomial Regression (Degree 2)
# ======================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate performance
poly_rmse = np.sqrt(mean_squared_error(y_test_poly, y_pred_poly))
poly_r2 = r2_score(y_test_poly, y_pred_poly)
print(f"Polynomial Regression (Degree 2) -> RMSE: {poly_rmse:.3f}, R²: {poly_r2:.3f}")

# ======================
# Model Performance Comparison
# ======================
print("\nModel Comparison:")
print(f"Linear Regression -> RMSE: {rmse:.3f}, R²: {r2:.3f}")
print(f"Polynomial Regression (Degree 2) -> RMSE: {poly_rmse:.3f}, R²: {poly_r2:.3f}")

# ======================
# Multiple Feature Regression
# ======================
features = ["hours_studied", "sleep_hours", "previous_score"]
available_features = [f for f in features if f in dataset.columns]

if available_features:
    X_multi = dataset[available_features]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

    multi_model = LinearRegression()
    multi_model.fit(X_train_m, y_train_m)
    y_pred_m = multi_model.predict(X_test_m)

    rmse_m = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
    r2_m = r2_score(y_test_m, y_pred_m)
    print(f"\nMultiple Feature Regression -> RMSE: {rmse_m:.3f}, R²: {r2_m:.3f}")

# ======================
# Visualizing Linear vs Polynomial Regression
# ======================
plt.figure(figsize=(7, 5))

# Scatter plot of actual data
plt.scatter(X, y, color="blue", alpha=0.6, label="Actual Data")

# Sort X for smooth curve plotting
X_sorted = np.sort(X.values, axis=0)

# Predictions for plotting
y_linear_pred_line = linear_model.predict(X_sorted)
y_poly_pred_line = poly_model.predict(poly.transform(X_sorted))

# Plot Linear Regression line
plt.plot(X_sorted, y_linear_pred_line, color="red", label="Linear Regression", linewidth=2)

# Plot Polynomial Regression curve
plt.plot(X_sorted, y_poly_pred_line, color="green", label="Polynomial Regression (deg=2)", linewidth=2)

# Labels and Title
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.show()
