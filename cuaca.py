import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Custom CSS for better styling
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .data-container {
        border-radius: 1px;
    }
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Set judul aplikasi
st.markdown('<div class="title">Visualisasi Data Time Series 🌤️</div>', unsafe_allow_html=True)

# Memuat data time series
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./Dataset/dataset time series.csv")
        # Mengganti nama kolom menjadi nama yang lebih deskriptif
        data.rename(columns={
            'TANGGAL': 'Tanggal',
            'TN': 'Suhu_Minimum',
            'TX': 'Suhu_Maksimum',
            'TAVG': 'Suhu_Rata_Rata',
            'RH_AVG': 'Kelembaban_Rata_Rata',
            'RR': 'Curah_Hujan',
            'SS': 'Sinar_Matahari',
            'FF_X': 'Kecepatan_Angin',
            'DDD_X': 'Arah_Angin',
            'FF_AVG': 'Kecepatan_Angin_Rata_Rata',
            'DDD_CAR': 'Deskripsi_Arah_Angin'
        }, inplace=True)
        
        # Mengubah kolom 'TN' dan 'RR' menjadi numerik, dengan error coercion
        data['Suhu_Minimum'] = pd.to_numeric(data['Suhu_Minimum'], errors='coerce')
        data['Curah_Hujan'] = pd.to_numeric(data['Curah_Hujan'], errors='coerce')
        
        # Mengubah kolom 'Tanggal' ke format datetime
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')
        # Ganti nilai 8888 dan 9999 dengan None (NaN)
        data = data.replace(({8888: None, 9999: None}))
        # menggantikan nilai yang hilang dengan nilai terdekat sebelumnya
        data = data.fillna(method='ffill')
        return data
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan path file benar.")
        return None

# Load data
time_series_data = load_data()

# Menampilkan data jika berhasil dimuat
if time_series_data is not None:
    # Membuat container dengan dua kolom yang lebih proporsional
    col1, col2 = st.columns([1, 2])
    
    # Menampilkan informasi dasar tentang dataset di kolom pertama
    with col1:
        st.markdown("<h3 style='text-align: center;'>🔍 Informasi Dataset</h3>", unsafe_allow_html=True)
        
        # Konversi tanggal ke format datetime untuk pencarian
        min_date = time_series_data['Tanggal'].min().date()
        max_date = time_series_data['Tanggal'].max().date()
        
        # Rentang tanggal dengan default rentang 10 hari dari tanggal minimum
        default_end_date = min_date + timedelta(days=10)
        if default_end_date > max_date:
            default_end_date = max_date
            
        date_range = st.date_input(
            "Pilih rentang tanggal",
            [min_date, default_end_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Flag untuk melacak apakah pencarian sedang aktif
        is_searching = False
        
        # Pilih kolom yang ditampilkan
        all_columns = time_series_data.columns.tolist()
        default_columns = ['Tanggal', 'Suhu_Minimum', 'Suhu_Maksimum', 'Curah_Hujan', 'Suhu_Rata_Rata', 'Kelembaban_Rata_Rata']
        selected_columns = st.multiselect(
            "Pilih kolom:",
            all_columns,
            default=default_columns
        )
        
        if not selected_columns:
            selected_columns = default_columns
        
        st.markdown(f"""
        <div class="data-container">
            <p>Jumlah baris: {time_series_data.shape[0]}</p>
            <p>Jumlah kolom: {time_series_data.shape[1]}</p>
        </div>
        """, unsafe_allow_html=True)
        # Tombol untuk melakukan pencarian - lebih jelas dan pada baris yang berbeda
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            search_button = st.button("Cari", use_container_width=True)
        with col_btn2:
            reset_button = st.button("Reset", use_container_width=True)
    
    # Menampilkan preview dataset di kolom kedua
    with col2:
        st.markdown("<h3 style='text-align: center;'>📋 Tampilan Dataset</h3>", unsafe_allow_html=True)
        
        filtered_data = time_series_data.copy()
        
        # Terapkan filter hanya jika tombol pencarian ditekan
        if search_button or is_searching:
            # Filter berdasarkan rentang tanggal yang dipilih
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = filtered_data[(filtered_data['Tanggal'].dt.date >= start_date) & 
                                            (filtered_data['Tanggal'].dt.date <= end_date)]
            # Tampilkan hasil filter
            if len(filtered_data) > 0:
                st.success(f"Ditemukan {len(filtered_data)} data yang sesuai kriteria pencarian.")
                st.dataframe(filtered_data[selected_columns], use_container_width=True)
            else:
                st.info("Tidak ada data yang sesuai dengan kriteria pencarian.")
        elif reset_button:
            # Reset pencarian
            filtered_data = time_series_data
            st.dataframe(filtered_data[selected_columns], use_container_width=True)
        else:
            # Tampilkan dataset dengan full width
            st.info("Gunakan fitur pencarian untuk melihat data spesifik.")
            st.dataframe(time_series_data[selected_columns], use_container_width=True)
            
    time_series_data['Hari'] = time_series_data['Tanggal'].dt.dayofweek
    time_series_data['Bulan'] = time_series_data['Tanggal'].dt.month

    def indonesia_season(month):
        if month in [11, 12, 1, 2, 3, 4]:
            return 'Musim Hujan'
        else:
            return 'Musim Kemarau'
        
    time_series_data['Musim'] = time_series_data['Bulan'].apply(indonesia_season)

# Menambahkan judul untuk bagian visualisasi
st.markdown('<div class="header">Visualisasi Pola Cuaca</div>', unsafe_allow_html=True)

# Tambahkan filter kalender dan filter tambahan untuk visualisasi
st.subheader("Filter Visualisasi")

# Buat tabs untuk memisahkan jenis filter
filter_tabs = st.tabs(["Filter Tanggal", "Filter Tambahan"])

with filter_tabs[0]:
    # Buat filter kalender dengan dua kolom
    col_cal1, col_cal2 = st.columns(2)

    with col_cal1:
        start_date = st.date_input(
            "Tanggal Mulai",
            time_series_data['Tanggal'].min().date(),
            min_value=time_series_data['Tanggal'].min().date(),
            max_value=time_series_data['Tanggal'].max().date()
        )
        
    with col_cal2:
        end_date = st.date_input(
            "Tanggal Akhir",
            time_series_data['Tanggal'].max().date(),
            min_value=time_series_data['Tanggal'].min().date(),
            max_value=time_series_data['Tanggal'].max().date()
        )

with filter_tabs[1]:
    # Filter tambahan dalam layout 2 kolom
    col_extra1, col_extra2 = st.columns(2)
    
    with col_extra1:
        # Filter musim
        selected_season = st.selectbox(
            "Pilih Musim:",
            ["Semua", "Musim Hujan", "Musim Kemarau"]
        )
        
        # Filter rentang suhu
        temp_min = float(time_series_data["Suhu_Rata_Rata"].min())
        temp_max = float(time_series_data["Suhu_Rata_Rata"].max())
        temp_range = st.slider(
            "Rentang Suhu Rata-Rata (°C):",
            temp_min,
            temp_max,
            (temp_min, temp_max)
        )
    
    with col_extra2:
        # Filter bulan
        month_options = list(range(1, 13))
        selected_months = st.multiselect(
            "Pilih Bulan:",
            month_options,
            default=month_options,
            format_func=lambda x: {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni", 
                               7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"}[x]
        )
        
        # Filter rentang curah hujan
        rain_min = float(time_series_data["Curah_Hujan"].min())
        rain_max = float(time_series_data["Curah_Hujan"].max())
        rain_range = st.slider(
            "Rentang Curah Hujan (mm):",
            rain_min,
            rain_max,
            (rain_min, rain_max)
        )

# Tombol untuk menerapkan filter
apply_filter = st.button("Terapkan Filter", key="apply_viz_filter", use_container_width=True)

# Fungsi untuk memproses data dengan filter
def filter_data(data):
    # Filter berdasarkan rentang tanggal
    filtered = data[(data['Tanggal'].dt.date >= start_date) & 
                    (data['Tanggal'].dt.date <= end_date)]
    
    # Filter tambahan
    if selected_season != "Semua":
        filtered = filtered[filtered['Musim'] == selected_season]
    
    if selected_months:
        filtered = filtered[filtered['Bulan'].isin(selected_months)]
    
    filtered = filtered[(filtered['Suhu_Rata_Rata'] >= temp_range[0]) & 
                     (filtered['Suhu_Rata_Rata'] <= temp_range[1])]
    
    filtered = filtered[(filtered['Curah_Hujan'] >= rain_range[0]) & 
                      (filtered['Curah_Hujan'] <= rain_range[1])]
    
    return filtered

# Tabs untuk setiap visualisasi
if apply_filter:
    # Filter data berdasarkan semua kriteria
    viz_data = filter_data(time_series_data)
    
    # Tampilkan informasi hasil filter
    if len(viz_data) > 0:
        st.success(f"Menampilkan data dari {start_date.strftime('%d-%m-%Y')} hingga {end_date.strftime('%d-%m-%Y')} ({len(viz_data)} data).")
        
        # Buat tabs untuk setiap visualisasi
        viz_tabs = st.tabs(["Suhu Rata-Rata", "Curah Hujan", "Kelembaban Rata-Rata", "Sinar Matahari"])
        
        # Tab Suhu Rata-Rata
        with viz_tabs[0]:
            st.subheader("Pola Suhu Rata-Rata")
            
            # Buat visualisasi Plotly
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=viz_data['Tanggal'],
                y=viz_data['Suhu_Rata_Rata'],
                mode='lines',
                name='Suhu Rata-Rata',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Styling
            fig_temp.update_layout(
                height=700,
                xaxis_title='Tanggal',
                yaxis_title='Suhu (°C)',
                title={
                    'text': 'Pola Suhu Rata-Rata',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                margin=dict(l=50, r=50, t=100, b=50),
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Statistik dasar
            st.subheader("Statistik Suhu Rata-Rata")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Suhu Rata-Rata", f"{viz_data['Suhu_Rata_Rata'].mean():.2f} °C")
            col_stat2.metric("Suhu Tertinggi", f"{viz_data['Suhu_Rata_Rata'].max():.2f} °C")
            col_stat3.metric("Suhu Terendah", f"{viz_data['Suhu_Rata_Rata'].min():.2f} °C")
            
        # Tab Curah Hujan
        with viz_tabs[1]:
            st.subheader("Pola Curah Hujan")
            
            # Buat visualisasi Plotly
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Scatter(
                x=viz_data['Tanggal'],
                y=viz_data['Curah_Hujan'],
                mode='lines',
                name='Curah Hujan',
                line=dict(color='blue', width=2)
            ))
            
            # Styling
            fig_rain.update_layout(
                height=700,  
                xaxis_title='Tanggal',
                yaxis_title='Curah Hujan (mm)',
                title={
                    'text': 'Pola Curah Hujan',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                margin=dict(l=50, r=50, t=100, b=50),
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_rain, use_container_width=True)
            
            # Statistik dasar
            st.subheader("Statistik Curah Hujan")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Curah Hujan Rata-Rata", f"{viz_data['Curah_Hujan'].mean():.2f} mm")
            col_stat2.metric("Curah Hujan Tertinggi", f"{viz_data['Curah_Hujan'].max():.2f} mm")
            col_stat3.metric("Curah Hujan Terendah", f"{viz_data['Curah_Hujan'].min():.2f} mm")
        
        # Tab Kelembaban Rata-Rata
        with viz_tabs[2]:
            st.subheader("Pola Kelembaban Rata-Rata")
            
            # Buat visualisasi Plotly
            fig_humid = go.Figure()
            fig_humid.add_trace(go.Scatter(
                x=viz_data['Tanggal'],
                y=viz_data['Kelembaban_Rata_Rata'],
                mode='lines',
                name='Kelembaban Rata-Rata',
                line=dict(color='green', width=2)
            ))
            
            # Styling
            fig_humid.update_layout(
                height=700,  # Full screen height
                xaxis_title='Tanggal',
                yaxis_title='Kelembaban (%)',
                title={
                    'text': 'Pola Kelembaban Rata-Rata',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                margin=dict(l=50, r=50, t=100, b=50),
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_humid, use_container_width=True)
            
            # Statistik dasar
            st.subheader("Statistik Kelembaban")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Kelembaban Rata-Rata", f"{viz_data['Kelembaban_Rata_Rata'].mean():.2f}%")
            col_stat2.metric("Kelembaban Tertinggi", f"{viz_data['Kelembaban_Rata_Rata'].max():.2f}%")
            col_stat3.metric("Kelembaban Terendah", f"{viz_data['Kelembaban_Rata_Rata'].min():.2f}%")
        
        # Tab Sinar Matahari
        with viz_tabs[3]:
            st.subheader("Pola Sinar Matahari")
            
            # Buat visualisasi Plotly
            fig_sun = go.Figure()
            fig_sun.add_trace(go.Scatter(
                x=viz_data['Tanggal'],
                y=viz_data['Sinar_Matahari'],
                mode='lines',
                name='Sinar Matahari',
                line=dict(color='orange', width=2)
            ))
            
            # Styling
            fig_sun.update_layout(
                height=700,  # Full screen height
                xaxis_title='Tanggal',
                yaxis_title='Sinar Matahari (Jam)',
                title={
                    'text': 'Pola Sinar Matahari',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                margin=dict(l=50, r=50, t=100, b=50),
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_sun, use_container_width=True)
            
            # Statistik dasar
            st.subheader("Statistik Sinar Matahari")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Durasi Rata-Rata", f"{viz_data['Sinar_Matahari'].mean():.2f} jam")
            col_stat2.metric("Durasi Terpanjang", f"{viz_data['Sinar_Matahari'].max():.2f} jam")
            col_stat3.metric("Durasi Terpendek", f"{viz_data['Sinar_Matahari'].min():.2f} jam")
            
    else:
        st.warning("Tidak ada data yang sesuai dengan filter yang diterapkan.")
else:
    # Tampilkan pesan awal
    st.info("Gunakan filter di atas dan klik 'Terapkan Filter' untuk melihat visualisasi.")    

# Menambahkan judul untuk bagian prediksi
st.markdown('<div class="header">Prediksi Suhu Menggunakan Machine Learning</div>', unsafe_allow_html=True)

# Misalnya, kita akan memprediksi suhu rata-rata (Suhu_Rata_Rata) berdasarkan fitur cuaca lainnya
features = ['Suhu_Minimum', 'Suhu_Maksimum', 'Kelembaban_Rata_Rata', 'Curah_Hujan', 'Sinar_Matahari', 'Kecepatan_Angin', 'Kecepatan_Angin_Rata_Rata', 'Bulan', 'Hari']
target = 'Suhu_Rata_Rata'

# Memisahkan fitur dan target
X = time_series_data[features]
y = time_series_data[target]

# 2. Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
col1, col2 = st.columns(2)

with col1:
    st.write("jumlah data x train: ", X_train.shape)
    st.write("jumlah data y train: ", y_train.shape)
with col2:
    st.write("jumlah data x test: ", X_test.shape)
    st.write("jumlah data y test: ", y_test.shape)

# 3. Memilih dan melatih model (Random Forest)
model = RandomForestRegressor(
    n_estimators=200,    # Meningkatkan jumlah pohon
    min_samples_split=10, # Meningkatkan minimum sampel untuk pembagian
    min_samples_leaf=5,  # Meningkatkan minimum sampel di daun
    random_state=42
)

model.fit(X_train, y_train)

# 4. Memprediksi menggunakan data pengujian
y_pred_rf = model.predict(X_test)

# Evaluasi Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

# Visualisasi hasil Random Forest dengan Plotly
st.subheader("Visualisasi Hasil Prediksi Random Forest")

# Buat DataFrame untuk plot
pred_df = pd.DataFrame({
    'Index': range(len(y_test)),
    'Actual': y_test.values,
    'Predicted': y_pred_rf
})

# Buat figure Plotly
fig = go.Figure()

# Tambahkan data aktual
fig.add_trace(go.Scatter(
    x=pred_df['Index'],
    y=pred_df['Actual'],
    mode='lines',
    name='Actual',
    line=dict(color='blue', width=2)
))

# Tambahkan data prediksi
fig.add_trace(go.Scatter(
    x=pred_df['Index'],
    y=pred_df['Predicted'],
    mode='lines',
    name='Predicted',
    line=dict(color='red', width=2, dash='dash')
))

# Update layout
fig.update_layout(
    height=600,
    xaxis_title='Index',
    yaxis_title='Suhu Rata-Rata (°C)',
    title={
        'text': 'Perbandingan Prediksi dan Aktual Suhu Rata-Rata (Random Forest)',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=20)
    },
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 0.8)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 0.8)'
    ),
    margin=dict(l=50, r=50, t=100, b=50),
    plot_bgcolor='white',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Tampilkan plot
st.plotly_chart(fig, use_container_width=True)

# Display evaluation metrics in two columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE (Rata-rata Kesalahan)", f"{mae_rf:.2f}")
    st.metric("MSE (Rata-rata Kesalahan Kuadrat)", f"{mse_rf:.2f}")
    
with col2:
    st.metric("RMSE (Akar Kesalahan Kuadrat)", f"{rmse_rf:.2f}")
    st.metric("MAPE (Kesalahan Persentase)", f"{mape_rf:.2f}%")

with col3:
    st.metric("R² (Akurasi Model)", f"{r2_rf:.2f} dari 1.00")

# Add footer
st.markdown("-----------")
st.markdown("""
<div style="text-align: center">
    <p>Visualisasi dan Prediksi Data Cuaca Rangkaian Waktu</p>
    <p>© 2025 - Mohammad Iqbal Maulana</p>
</div>
""", unsafe_allow_html=True)