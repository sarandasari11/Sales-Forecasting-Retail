import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Retail Sales Forecast Dashboard", layout="wide")

from pathlib import Path

# ----------------------------
# Path Configuration
# ----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if (CURRENT_DIR.parent / "data" / "sales.csv").exists() else CURRENT_DIR

DATA_PATH = PROJECT_ROOT / "data" / "sales.csv"
XGB_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_model.pkl"
LSTM_PATH = PROJECT_ROOT / "outputs" / "models" / "lstm_model.h5"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# ----------------------------
# Load trained models
# ----------------------------
xgb_model = joblib.load(XGB_PATH)
lstm_model = load_model(LSTM_PATH, compile=False)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Settings")
store_list = df['Store'].unique()
store_id = st.sidebar.selectbox("Select Store", store_list)

model_options = st.sidebar.multiselect(
    "Select models to display", ['XGBoost','LSTM','ARIMA'], default=['XGBoost','LSTM','ARIMA']
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['Date'].min(), df['Date'].max()]
)

current_stock = st.sidebar.number_input("Enter Current Stock", min_value=0, value=1000)

# ----------------------------
# Filter store data
# ----------------------------
store_data = df[df['Store'] == store_id].sort_values('Date')
store_data = store_data[(store_data['Date'] >= pd.to_datetime(date_range[0])) &
                        (store_data['Date'] <= pd.to_datetime(date_range[1]))]

if store_data.empty:
    st.warning("No data available for the selected store/date range!")
    st.stop()

if 'Weekly_Sales' not in store_data.columns:
    st.error("Column 'Weekly_Sales' not found in the dataset!")
    st.stop()

# ----------------------------
# Feature Engineering
# ----------------------------
if len(store_data) > 4:
    store_data['Month'] = store_data['Date'].dt.month
    store_data['Week'] = store_data['Date'].dt.isocalendar().week
    store_data['Lag_1'] = store_data['Weekly_Sales'].shift(1)
    store_data['Lag_2'] = store_data['Weekly_Sales'].shift(2)
    store_data['Rolling_4'] = store_data['Weekly_Sales'].shift(1).rolling(4).mean()
    store_data = store_data.dropna()
else:
    st.error("Not enough data to create lag features!")
    st.stop()

features = ['Month', 'Week', 'Lag_1', 'Lag_2', 'Rolling_4']

# ----------------------------
# Predictions
# ----------------------------
y_xgb_pred, y_lstm_pred, y_arima_pred = None, None, None
xgb_dates, lstm_dates, arima_dates = None, None, None

# XGBoost
if 'XGBoost' in model_options:
    y_xgb_pred = xgb_model.predict(store_data[features].astype(float))
    xgb_dates = store_data['Date']

# LSTM
if 'LSTM' in model_options:
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(store_data[['Weekly_Sales']].values)
    time_steps = 10
    X_lstm = np.array([sales_scaled[i:(i+time_steps),0] for i in range(len(sales_scaled)-time_steps)])
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1],1))
    y_lstm_pred = scaler.inverse_transform(lstm_model.predict(X_lstm))
    lstm_dates = store_data['Date'].iloc[time_steps:]

# ARIMA
if 'ARIMA' in model_options:
    ts = store_data.set_index('Date')['Weekly_Sales']
    arima_model = ARIMA(ts, order=(2,1,2))
    arima_fit = arima_model.fit()
    y_arima_pred = arima_fit.fittedvalues[1:]
    arima_dates = ts.index[1:]

# ----------------------------
# Metrics (RMSE & MAE)
# ----------------------------
metrics_data = []

def compute_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return [model_name, rmse, mae]

if y_xgb_pred is not None:
    metrics_data.append(compute_metrics(store_data['Weekly_Sales'], y_xgb_pred, "XGBoost"))

if y_lstm_pred is not None:
    y_true_lstm = store_data['Weekly_Sales'].iloc[-len(y_lstm_pred):]
    metrics_data.append(compute_metrics(y_true_lstm, y_lstm_pred.flatten(), "LSTM"))

if y_arima_pred is not None:
    y_true_arima = store_data['Weekly_Sales'].iloc[-len(y_arima_pred):]
    metrics_data.append(compute_metrics(y_true_arima, y_arima_pred, "ARIMA"))

metrics_df = pd.DataFrame(metrics_data, columns=['Model','RMSE','MAE']).round(2)

# Highlight best model
best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model'] if not metrics_df.empty else None
def highlight_best(row):
    return ['background-color: green' if row['Model'] == best_model else '' for _ in row]

# ----------------------------
# Forecasted stock recommendation
# ----------------------------
forecasted_sales = None
if y_xgb_pred is not None:
    forecasted_sales = y_xgb_pred[-1]
elif y_lstm_pred is not None:
    forecasted_sales = y_lstm_pred.flatten()[-1]
elif y_arima_pred is not None:
    forecasted_sales = y_arima_pred[-1]

if forecasted_sales is not None:
    forecasted_sales = float(forecasted_sales)
    recommended_stock = forecasted_sales * 1.1
    if current_stock < recommended_stock:
        stock_status, stock_color = "⚠️ Understock Risk", "red"
    elif current_stock > recommended_stock * 1.2:
        stock_status, stock_color = "⚠️ Overstock Risk", "orange"
    else:
        stock_status, stock_color = "✅ Stock OK", "green"
else:
    recommended_stock, stock_status, stock_color = None, "No forecast available", "grey"



# ----------------------------
# Tabs layout
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Forecast Comparison", "📥 Download Data"])

# ----------------------------
# Tab1: Summary
# ----------------------------
with tab1:
    st.subheader("Store Summary")
    st.write(f"**Store Selected:** {store_id}")
    st.write(f"**Date Range:** {store_data['Date'].min().date()} to {store_data['Date'].max().date()}")
    st.write(f"**Forecasted Sales:** {forecasted_sales:.0f}" if forecasted_sales else "N/A")
    st.write(f"**Recommended Stock:** {recommended_stock:.0f}" if recommended_stock else "N/A")
    
    st.markdown(f"<h3 style='color:{stock_color}'>{stock_status}</h3>", unsafe_allow_html=True)

    # Stock comparison bar chart
    if recommended_stock is not None:
        stock_df = pd.DataFrame({'Type': ['Current Stock', 'Recommended Stock'],
                                 'Units': [current_stock, recommended_stock]})
        fig_stock = px.bar(stock_df, x='Units', y='Type', orientation='h',
                           text='Units', color='Type',
                           color_discrete_map={'Current Stock':'blue', 'Recommended Stock':'green'})
        fig_stock.update_layout(title="Stock Recommendation vs Current Stock",
                                xaxis_title="Units", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_stock, use_container_width=True)

    st.markdown("### Forecast Accuracy Metrics")
    col1, col2, col3 = st.columns(3)
    for i, row in metrics_df.iterrows():
        col = [col1, col2, col3][i%3]
        col.metric(label=row['Model'], value=f"RMSE: {row['RMSE']:.2f}", delta=f"MAE: {row['MAE']:.2f}")
# ----------------------------
# Tab2: Forecast Comparison
# ----------------------------
with tab2:
    st.subheader("Forecast vs Actual Sales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=store_data['Date'], y=store_data['Weekly_Sales'], mode='lines', name='Actual Sales'))
    if y_xgb_pred is not None:
        fig.add_trace(go.Scatter(x=xgb_dates, y=y_xgb_pred, mode='lines', name='XGBoost Forecast'))
    if y_lstm_pred is not None:
        fig.add_trace(go.Scatter(x=lstm_dates, y=y_lstm_pred.flatten(), mode='lines', name='LSTM Forecast'))
    if y_arima_pred is not None:
        fig.add_trace(go.Scatter(x=arima_dates, y=y_arima_pred, mode='lines', name='ARIMA Forecast'))
    fig.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Forecast Accuracy Metrics
    st.markdown("### Forecast Accuracy Metrics")
    if not metrics_df.empty:
        st.dataframe(metrics_df.style.apply(highlight_best, axis=1), use_container_width=True)
# ----------------------------
# Tab3: Download Data
# ----------------------------
with tab3:
    st.subheader("Download Forecast Data")
    forecast_df = pd.DataFrame({'Date': store_data['Date'].reset_index(drop=True),
                                'Actual_Sales': store_data['Weekly_Sales'].reset_index(drop=True)})
    if y_xgb_pred is not None: forecast_df['XGBoost_Forecast'] = y_xgb_pred
    if y_lstm_pred is not None: forecast_df['LSTM_Forecast'] = np.concatenate(([np.nan]*(len(store_data)-len(y_lstm_pred)), y_lstm_pred.flatten()))
    if y_arima_pred is not None: forecast_df['ARIMA_Forecast'] = np.concatenate(([np.nan]*(len(store_data)-len(y_arima_pred)), y_arima_pred))
    
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button(label="📥 Download Forecast CSV", data=csv,
                       file_name=f"store_{store_id}_forecast.csv", mime="text/csv")
