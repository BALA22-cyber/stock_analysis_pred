import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the data and inspect it
# Load without assumptions first to check column names
# data = pd.read_csv('NIFTY50.csv')
# print("Column names in the CSV file:", data.columns.tolist())

# # Assuming the first column is the date column, adjust based on inspection
# date_column = data.columns[0]  # Dynamically use the first column as the date
# data[date_column] = pd.to_datetime(data[date_column])  # Convert to datetime
# data.set_index(date_column, inplace=True)

data = pd.read_csv('NIFTY50_max_data.csv', skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'], parse_dates=['Date'], index_col='Date')


# Drop unnecessary columns (assuming Price and Ticker are present but not needed)
data = data[['Close', 'High', 'Low', 'Open', 'Volume']]

# Step 2: Identify the first trading day of each week
data['Week'] = data.index.isocalendar().week
data['Year'] = data.index.isocalendar().year
data['Week_Change'] = (data['Week'] != data['Week'].shift(1)) | (data['Year'] != data['Year'].shift(1))
data.loc[data.index[0], 'Week_Change'] = True
data['Is_First_Trading_Day'] = data['Week_Change'].astype(int)

# Step 3: Compute the gap and assign labels
data['Prev_Close'] = data['Close'].shift(1)
data['Gap'] = (data['Open'] - data['Prev_Close']) / data['Prev_Close']
threshold = 0.002

def assign_label(gap):
    if pd.isna(gap):
        return np.nan
    elif gap > threshold:
        return 2  # Gap Up
    elif gap < -threshold:
        return 0  # Gap Down
    else:
        return 1  # Neutral

data['Label'] = data['Gap'].apply(assign_label)
prediction_days = data[data['Is_First_Trading_Day'] == 1].copy()

# Step 4: Feature Engineering
N = 5
X = []
y = []

for idx in prediction_days.index:
    idx_loc = data.index.get_loc(idx)
    if idx_loc >= N:
        feature_vector = data.iloc[idx_loc - N:idx_loc][['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        label = prediction_days.loc[idx, 'Label']
        if not pd.isna(label):
            X.append(feature_vector)
            y.append(label)

X = pd.DataFrame(X, columns=[f'{col}_{i}' for i in range(N) for col in ['Open', 'High', 'Low', 'Close', 'Volume']])
y = pd.Series(y)

# Step 5: Split the data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=10000, random_state=42)
# model = SVC(kernel='rbf', random_state=42)
# model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Gap Down', 'Neutral', 'Gap Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Predict on the last few days
last_few_features = X.tail(10)
last_few_predictions = model.predict(last_few_features)
print("\nPredictions for the last 5 first-trading-days:")
for date, pred in zip(data[data['Is_First_Trading_Day'] == 1].index[-10:], last_few_predictions):
    label = {0: 'Gap Down', 1: 'Neutral', 2: 'Gap Up'}[pred]
    print(f"{date.date()}: {label}")