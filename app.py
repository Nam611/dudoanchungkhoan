import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yfinance as yf

# Đặt cấu hình trang
st.set_page_config(
    page_title="Dự đoán giá chứng khoán",
    layout="wide"
)

# Thêm tiêu đề và mô tả
st.title("Dự đoán giá chứng khoán")
st.write("Ứng dụng này dự đoán giá chứng khoán bằng mô hình LSTM.")

# Đường dẫn mô hình
model_paths = {
    "Open": "btc_model_Open4.h5",
    "High": "btc_model_High2.h5",
    "Low": "btc_model_Low5.h5",
    "Close": "btc_model_Close5.h5"
}

# Hàm để tải dữ liệu chứng khoán từ yfinance
def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    df.reset_index(inplace=True)
    return df

# Đầu vào thanh bên cho phạm vi ngày tùy chỉnh dưới dạng văn bản
st.sidebar.write("### Nhập ngày bắt đầu và ngày kết thúc (YYYY-MM-DD)")
start_date_input = st.sidebar.text_input("Ngày bắt đầu", value="2023-01-01")
end_date_input = st.sidebar.text_input("Ngày kết thúc", value=datetime.today().strftime('%Y-%m-%d'))

# Danh sách mã cổ phiếu để người dùng lựa chọn
st.sidebar.write("### Chọn mã cổ phiếu")
stock_ticker = st.sidebar.selectbox(
    "Chọn mã cổ phiếu để tải dữ liệu",
    ("BTC-USD", "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN","META","VFS")
)

# Xử lý ngày hợp lệ
try:
    # Chuyển đổi chuỗi đầu vào thành đối tượng datetime
    start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_input, '%Y-%m-%d')
    
    # Xác thực rằng end_date không trước start_date
    if start_date >= end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu.")
    else:
        # Tìm nạp dữ liệu cho mã cổ phiếu đã chọn và phạm vi ngày
        df = fetch_stock_data(ticker=stock_ticker, start_date=start_date_input, end_date=end_date_input)
except ValueError:
    st.error("Vui lòng nhập ngày hợp lệ theo định dạng YYYY-MM-DD.")

# Tải dữ liệu và cho phép tải lên thủ công hoặc tự động tìm nạp
data_source = st.sidebar.radio("Chọn nguồn dữ liệu:", ("Tự động cập nhật từ yfinance", "Tải lên tệp CSV"))

if data_source == "Tự động cập nhật từ yfinance":
    df = fetch_stock_data(ticker=stock_ticker, start_date=start_date_input, end_date=end_date_input)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV dữ liệu", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.info("Vui lòng tải lên tệp CSV hoặc chọn tự động cập nhật từ yfinance.")
        st.stop()
# Hiển thị tổng quan dữ liệu
st.subheader("Tổng quan về dữ liệu")
st.dataframe(df.head())

# Thanh bên để lựa chọn loại giá
price_type = st.sidebar.selectbox(
    "Chọn loại giá để dự đoán",
    ['Open', 'High', 'Low', 'Close']
)

# Vẽ giá lịch sử cho loại đã chọn
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

# Phần dự đoán
st.subheader(f"{price_type} Dự đoán giá")
days_to_predict = st.slider("Chọn số ngày để dự đoán", 1, 30, 7)

if st.button("Dự đoán"):
    try:
        # Tải mô hình cho loại giá đã chọn
        model_path = model_paths.get(price_type)
        if model_path:
            model = load_model(model_path)
        else:
            st.error(f"No model available for {price_type} price.")
            st.stop()
        
        # Chuẩn bị dữ liệu để dự đoán
        look_back = 60
        last_60_days = df[price_type].values[-look_back:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        last_60_days_scaled = scaler.fit_transform(last_60_days.reshape(-1, 1))
        
        # Đưa ra dự đoán
        predictions = []
        current_batch = last_60_days_scaled.reshape((1, look_back, 1))
        
        for i in range(days_to_predict):
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Dự đoán biến đổi nghịch đảo
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Tạo ngày trong tương lai
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(days_to_predict)]
        
        # Cốt truyện dự đoán
        fig_pred = go.Figure()
        
        # Giá lịch sử
        fig_pred.add_trace(go.Scatter(
            x=df['Date'][-100:],
            y=df[price_type][-100:],
            name=f'Historical {price_type} Price',
            line=dict(color='blue')
        ))
        
        # Dự đoán
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
           
        # Thêm số liệu thống kê bổ sung
        st.subheader("Thống kê dự đoán")
        col1, col2, col3 = st.columns(3)

        
        with col1:
            st.metric(
                label="Giá hiện tại",
                value=f"${float(df[price_type].iloc[-1]):,.2f}"  # Rõ ràng đúc để nổi
            )
        
        with col2:
            st.metric(
                label="Dự đoán giá cuối cùng",
                value=f"${float(predictions[-1][0]):,.2f}",  # Rõ ràng đúc để nổi
                delta=f"{((float(predictions[-1][0]) - float(df[price_type].iloc[-1])) / float(df[price_type].iloc[-1]) * 100):,.2f}%"  # Explicitly cast to float
            )
        
        with col3:
            st.metric(
                label="Giá tối đa dự đoán",
                value=f"${float(np.max(predictions)):,.2f}"  # Rõ ràng đúc để nổi
            )
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Thêm chân trang
st.markdown("""
---
### Notes:
- Dự đoán dựa trên dữ liệu lịch sử và không nên được sử dụng làm tư vấn tài chính
- Mô hình sử dụng dữ liệu của 60 ngày qua để đưa ra dự đoán
- Dự đoán trở nên kém chính xác hơn khi thời gian dự đoán tăng lên
""")
