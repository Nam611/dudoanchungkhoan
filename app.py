import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="Dự đoán giá Bitcoin",
    layout="wide"
)

# Add title and description
st.title("Dự đoán giá Bitcoin")
st.write("Ứng dụng này dự đoán giá Bitcoin bằng mô hình LSTM.")

# Model paths
model_paths = {
    "Open": "btc_model_Open.h5",
    "High": "btc_model_High.h5",
    "Low": "btc_model_Low.h5",
    "Close": "btc_model_Close.h5"
}

# Function to fetch BTC-USD data from yfinance for a custom date range
@st.cache_data
def fetch_btc_data(start_date: str, end_date: str):
    df = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    df.reset_index(inplace=True)
    return df

# Sidebar inputs for custom date range as text
st.sidebar.write("### Nhập ngày bắt đầu và ngày kết thúc (YYYY-MM-DD)")
start_date_input = st.sidebar.text_input("Ngày bắt đầu", value="2023-01-01")
end_date_input = st.sidebar.text_input("Ngày kết thúc", value=datetime.today().strftime('%Y-%m-%d'))

try:
    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_input, '%Y-%m-%d')
    
    # Validate that end_date is not before start_date
    if start_date >= end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu.")
    else:
        # Fetch data for the selected date range
        df = fetch_btc_data(start_date=start_date_input, end_date=end_date_input)
except ValueError:
    st.error("Vui lòng nhập ngày hợp lệ theo định dạng YYYY-MM-DD.")

# Load data and allow for manual upload or auto-fetch
data_source = st.sidebar.radio("Chọn nguồn dữ liệu:", ("Tự động cập nhật từ yfinance", "Tải lên tệp CSV"))

if data_source == "Tự động cập nhật từ yfinance":
    df = fetch_btc_data(start_date=start_date_input, end_date=end_date_input)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV dữ liệu", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.info("Vui lòng tải lên tệp CSV hoặc chọn tự động cập nhật từ yfinance.")
        st.stop()

# Display data overview
st.subheader("Tổng quan về dữ liệu")
st.dataframe(df.head())

# Sidebar for price type selection
price_type = st.sidebar.selectbox(
    "Chọn loại giá để dự đoán",
    ['Open', 'High', 'Low', 'Close']
)

# Plot historical prices for selected type
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df[price_type],
    name=f'Historical {price_type} Price',
    line=dict(color='blue')
))
fig.update_layout(
    title=f'Lịch sử Bitcoin {price_type} Prices',
    xaxis_title='Date',
    yaxis_title='Giá (USD)',
    template='plotly_white'
)
st.plotly_chart(fig)

# Prediction section
st.subheader(f"{price_type} Dự đoán giá")
days_to_predict = st.slider("Chọn số ngày để dự đoán", 1, 30, 7)

if st.button("Dự đoán"):
    try:
        # Load the model for the selected price type
        model_path = model_paths.get(price_type)
        if model_path:
            model = load_model(model_path)
        else:
            st.error(f"No model available for {price_type} price.")
            st.stop()
        
        # Prepare data for prediction
        look_back = 60
        last_60_days = df[price_type].values[-look_back:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        last_60_days_scaled = scaler.fit_transform(last_60_days.reshape(-1, 1))
        
        # Make predictions
        predictions = []
        current_batch = last_60_days_scaled.reshape((1, look_back, 1))
        
        for i in range(days_to_predict):
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(days_to_predict)]
        
        # Plot predictions
        fig_pred = go.Figure()
        
        # Historical prices
        fig_pred.add_trace(go.Scatter(
            x=df['Date'][-100:],
            y=df[price_type][-100:],
            name=f'Historical {price_type} Price',
            line=dict(color='blue')
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=future_dates,
            y=predictions.flatten(),
            name=f'Predicted {price_type} Price',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f'Bitcoin {price_type} Dự đoán giá',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pred)
        
        with st.expander(f"Dự đoán {price_type} Giá"):
            pred_df = pd.DataFrame({
                'Date': future_dates,
               f'Predicted {price_type} Price': predictions.flatten()
            })
            st.dataframe(pred_df)
           
        # Add additional statistics
        st.subheader("Thống kê dự đoán")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Giá hiện tại",
                value=f"${df[price_type].iloc[-1]:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Dự đoán giá cuối cùng",
                value=f"${predictions[-1][0]:,.2f}",
                delta=f"{((predictions[-1][0] - df[price_type].iloc[-1])/df[price_type].iloc[-1]*100):,.2f}%"
            )
        
        with col3:
            st.metric(
                label="Giá tối đa dự đoán",
                value=f"${np.max(predictions):,.2f}"
            )
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("""
---
### Notes:
- Dự đoán dựa trên dữ liệu lịch sử và không nên được sử dụng làm tư vấn tài chính
- Mô hình sử dụng dữ liệu của 60 ngày qua để đưa ra dự đoán
- Dự đoán trở nên kém chính xác hơn khi thời gian dự đoán tăng lên
""")
