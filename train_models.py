import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales.csv"
MODELS_DIR = BASE_DIR / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# -----------------------------
# 1. Train and Save XGBoost Model
# -----------------------------
print("Training XGBoost model...")
df_features = df.copy()
df_features['Month'] = df_features['Date'].dt.month
df_features['Week'] = df_features['Date'].dt.isocalendar().week
df_features['Lag_1'] = df_features.groupby('Store')['Weekly_Sales'].shift(1)
df_features['Lag_2'] = df_features.groupby('Store')['Weekly_Sales'].shift(2)
df_features['Rolling_4'] = df_features.groupby('Store')['Weekly_Sales'].shift(1).rolling(4).mean()

df_clean = df_features.dropna()
features = ['Month', 'Week', 'Lag_1', 'Lag_2', 'Rolling_4']
target = 'Weekly_Sales'

X = df_clean[features].astype(float)
y = df_clean[target].astype(float)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X, y)

xgb_path = MODELS_DIR / "xgb_model.pkl"
joblib.dump(xgb_model, xgb_path)
print(f"Saved XGBoost model to {xgb_path}")

# -----------------------------
# 2. Train and Save LSTM Model
# -----------------------------
print("Training LSTM model...")
# Train on overall sales / store sales sequence
sales_values = df[['Weekly_Sales']].values.astype(np.float32)
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales_values)

time_steps = 10
X_lstm, y_lstm = [], []
for i in range(len(sales_scaled) - time_steps):
    X_lstm.append(sales_scaled[i:(i + time_steps), 0])
    y_lstm.append(sales_scaled[i + time_steps, 0])

X_lstm = np.array(X_lstm).reshape((-1, time_steps, 1))
y_lstm = np.array(y_lstm).reshape((-1, 1))

lstm_model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=64, verbose=1)

lstm_path = MODELS_DIR / "lstm_model.h5"
lstm_model.save(lstm_path)
print(f"Saved LSTM model to {lstm_path}")

print("All models trained and saved successfully!")
