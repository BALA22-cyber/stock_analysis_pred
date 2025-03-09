import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier#,xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
# file_path = "NIFTY50.csv"  # Update with the correct path if needed
# data = pd.read_csv(file_path)

# # Fix column headers
# data.columns = data.iloc[0]
# data = data[1:].reset_index(drop=True)
# data.rename(columns={'Price': 'Date'}, inplace=True)
# # data = data.loc[:, ~data.columns.str.contains("Unnamed")]
# # print(data.head(),data.shape)
# # Convert data types
# data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
# numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
# data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# # Drop NaN values
# data = data.dropna()

file_path = "NIFTY50.csv"
# Skip the first two rows (metadata) and assign proper column names
data = pd.read_csv(file_path, skiprows=2, header=None, 
                   names=["Date", "Close", "High", "Low", "Open", "Volume"])
data = data.reset_index(drop=True)

# Convert Date column
data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

# Print column names to debug
print("Column names:", data.columns.tolist())

# Convert numeric columns
numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop NaN values
data = data.dropna()

# Ensure proper column names are present
if not all(col in data.columns for col in numeric_columns):
    print("Warning: Missing expected columns!")
    print("Available columns:", data.columns.tolist())

# Extract weekday information
data["Weekday"] = data["Date"].dt.weekday  # 0=Monday, 4=Friday

# Create shifted columns for previous week's closing prices
data["Close_Mon"] = data["Close"].shift(5).where(data["Weekday"] == 0)
data["Close_Tue"] = data["Close"].shift(4).where(data["Weekday"] == 0)
data["Close_Wed"] = data["Close"].shift(3).where(data["Weekday"] == 0)
data["Close_Thu"] = data["Close"].shift(2).where(data["Weekday"] == 0)
data["Close_Fri"] = data["Close"].shift(1).where(data["Weekday"] == 0)

# Compute additional features
data["Weekly_Volatility"] = (data["Close_Fri"] - data["Close_Mon"]).abs()
data["Average_Weekly_Close"] = data[["Close_Mon", "Close_Tue", "Close_Wed", "Close_Thu", "Close_Fri"]].mean(axis=1)
data["Friday_Volume"] = data["Volume"].shift(1).where(data["Weekday"] == 0)

# Extract only Monday rows for prediction
mondays = data[data["Weekday"] == 0].copy()

# Ensure Friday’s close is correctly mapped
mondays["Prev_Friday_Close"] = mondays["Close_Fri"].fillna(method="ffill")

# Define the target variable (Gap Type)
threshold = 0.005  # 0.5% threshold
# mondays["Gap_Type"] = mondays.apply(
#     lambda row: 1 if row["Open"] > row["Prev_Friday_Close"] * (1 + threshold) else 
#                 -1 if row["Open"] < row["Prev_Friday_Close"] * (1 - threshold) else 0, 
#     axis=1
# )

mondays["Gap_Type"] = mondays.apply(
    lambda row: 2 if row["Open"] > row["Prev_Friday_Close"] * (1 + threshold) else 
                0 if row["Open"] < row["Prev_Friday_Close"] * (1 - threshold) else 1, 
    axis=1
)

# Define the feature set for training
feature_columns = ["Close_Mon", "Close_Tue", "Close_Wed", "Close_Thu", "Close_Fri", 
                   "Weekly_Volatility", "Average_Weekly_Close", "Friday_Volume"]
X = mondays[feature_columns].dropna()
y = mondays["Gap_Type"].dropna()

# Ensure consistency between X and y
X = X.dropna()
y = y.loc[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train the Random Forest Classifier
# model = RandomForestClassifier(n_estimators=10000, random_state=42)
model = xgb.XGBClassifier(n_estimators=100000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
