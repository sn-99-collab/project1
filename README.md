import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/content/archive (3).zip')

# Handle missing values
df_cleaned = df.dropna()

# Define the target variable y and predictor variables X
y = df_cleaned['mpg']
X = df_cleaned.drop(['mpg', 'car name'], axis=1)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on test data
y_pred = lin_reg.predict(X_test)

# Model evaluation
print("Linear Regression Model:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
print()

# Polynomial Regression model
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_scaled)

X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_poly_train)

# Predict on test data
y_poly_pred = poly_reg.predict(X_poly_test)

# Model evaluation
print("Polynomial Regression Model (degree=2):")
print("Mean Squared Error:", mean_squared_error(y_poly_test, y_poly_pred))
print("R-squared:", r2_score(y_poly_test, y_poly_pred))
print()

# Data visualization (as shown previously)
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['mpg'], bins=20, kde=True)
plt.title('Distribution of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(df_cleaned[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = df_cleaned[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='cylinder', y='mpg', data=df_cleaned)
plt.title('MPG across different Cylinder values')
plt.xlabel('Cylinder')
plt.ylabel('MPG')
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x='model year', y='mpg', data=df_cleaned)
plt.title('MPG across different Model Year values')
plt.xlabel('Model Year')
plt.ylabel('MPG')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='origin', y='mpg', data=df_cleaned)
plt.title('MPG across different Origin values')
plt.xlabel('Origin')
plt.ylabel('MPG')
plt.show()
