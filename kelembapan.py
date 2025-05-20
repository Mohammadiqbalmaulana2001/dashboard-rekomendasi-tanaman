import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


#⭕ """"ini tampilan informasi dataset"""" ⭕

st.markdown("<h1 style='text-align: center;'>💧Peramalan Kelembapan Rata-Rata 💧</h1>", unsafe_allow_html=True)
# Deskripsi
st.markdown("""
<div style='text-align: justify; font-size: 16px; text-indent: 40px; padding-bottom: 50px;'>
    Dalam proyek ini, kita akan menganalisis data time series cuaca untuk memprediksi kelembapan rata-rata (<code>RH_AVG</code>) berdasarkan berbagai parameter cuaca. 
    Tujuan utama adalah membangun model machine learning seperti <b>Random Forest</b> yang dapat memperkirakan kelembapan dengan akurat berdasarkan fitur-fitur seperti suhu, curah hujan, dan sinar matahari.
</div>
""", unsafe_allow_html=True)
# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./Dataset/dataset time series.csv")
        
        # Mengkonversi kolom bertipe object ke numerik
        for col in ["TN", "RR"]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Mengubah kolom 'TANGGAL' ke format datetime
        data['TANGGAL'] = pd.to_datetime(data['TANGGAL'], format='%d-%m-%Y')
        
        # Ganti nilai 8888 dan 9999 dengan NaN
        data = data.replace({8888: np.nan, 9999: np.nan})
        
        # Mengisi nilai yang hilang dengan interpolasi
        data = data.interpolate(method='linear')
        
        # Mengubah nama kolom sesuai dengan deskripsi yang lebih mudah dipahami
        data = data.rename(columns={
            'TANGGAL': 'Tanggal',
            'TN': 'Suhu_Minimum',
            'TX': 'Suhu_Maksimum',
            'RH_AVG': 'Kelembaban_Rata_Rata',
            'RR': 'Curah_Hujan',
            'SS': 'Sinar_Matahari',
            'FF_X': 'Kecepatan_Angin_Max',
            'FF_AVG': 'Kecepatan_Angin_Rata_Rata',
            'TAVG': 'Suhu_Rata_Rata',
            'DDD_X': 'Arah_Angin',
            'DDD_CAR': 'Deskripsi_Arah_Angin'
        })
        
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






# 🤖 Visualisasi Pola Cuaca 🤖

st.markdown("---")
st.markdown("<h2 style='text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>📊 Visualisasi Pola Cuaca 📊</h2>", unsafe_allow_html=True)

# Filter tanggal untuk visualisasi
col1, col2 = st.columns(2)
with col1:
    start_date_vis = st.date_input(
        "Tanggal Mulai",
        value=time_series_data['Tanggal'].min(),
        min_value=time_series_data['Tanggal'].min(),
        max_value=time_series_data['Tanggal'].max()
    )
with col2:
    end_date_vis = st.date_input(
        "Tanggal Akhir",
        value=time_series_data['Tanggal'].max(),
        min_value=time_series_data['Tanggal'].min(),
        max_value=time_series_data['Tanggal'].max()
    )

# Filter data berdasarkan tanggal yang dipilih
filtered_vis_data = time_series_data[
    (time_series_data['Tanggal'] >= pd.to_datetime(start_date_vis)) &
    (time_series_data['Tanggal'] <= pd.to_datetime(end_date_vis))
]

# Menghitung periode waktu untuk info
days_count = (pd.to_datetime(end_date_vis) - pd.to_datetime(start_date_vis)).days + 1
st.info(f"Menampilkan {days_count} data untuk periode {start_date_vis} hingga {end_date_vis}")

# Membuat tab
tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Suhu Rata-Rata", 
    "🌧️ Curah Hujan",
    "💧 Kelembaban Rata-Rata", 
    "☀️ Sinar Matahari"
])

# Style untuk statistik cards dengan angka besar dan teks tengah
stats_style = """
<style>
    .stat-card {
        border-radius: 10px;
        padding: 20px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 100%;
        text-align: center;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .stat-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
        color: #555;
        text-align: center;
    }
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        margin: 15px 0;
        text-align: center;
    }
    .stat-unit {
        font-size: 16px;
        color: #666;
        text-align: center;
    }
    .interpretation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 0 5px 5px 0;
        margin-top: 20px;
    }
</style>
"""
st.markdown(stats_style, unsafe_allow_html=True)

with tab1:
    st.markdown("<h3 style='text-align: center; color: #E53935;'>Pola Suhu Rata-Rata</h3>", unsafe_allow_html=True)
    
    # Membuat grafik
    fig1 = px.line(
        filtered_vis_data,
        x='Tanggal',
        y='Suhu_Rata_Rata',
        labels={'Suhu_Rata_Rata': 'Suhu'},
        color_discrete_sequence=['#E53935']
    )
    fig1.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        xaxis_title='Tanggal',
        yaxis_title="Suhu (°C)",
        margin=dict(t=40, b=20),
        xaxis=dict(
            tickformat='%d %b %Y',
            tickangle=-45
        )
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Statistik suhu
    temp_mean = filtered_vis_data['Suhu_Rata_Rata'].mean()
    temp_max = filtered_vis_data['Suhu_Rata_Rata'].max()
    temp_min = filtered_vis_data['Suhu_Rata_Rata'].min()
    
    st.markdown("<h4 style='text-align: center; color: #E53935; margin-top: 20px;'>Statistik Suhu</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #ffebee; border-top: 3px solid #E53935;">
            <div class="stat-title">Rata-Rata</div>
            <div class="stat-value" style="color: #E53935;">{temp_mean:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #ffebee; border-top: 3px solid #E53935;">
            <div class="stat-title">Tertinggi</div>
            <div class="stat-value" style="color: #E53935;">{temp_max:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #ffebee; border-top: 3px solid #E53935;">
            <div class="stat-title">Terendah</div>
            <div class="stat-value" style="color: #E53935;">{temp_min:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h3 style='text-align: center; color: #0000ff;'>Pola Curah Hujan</h3>", unsafe_allow_html=True)
    
    # Membuat grafik
    fig2 = px.line(
        filtered_vis_data,
        x='Tanggal',
        y='Curah_Hujan',
        labels={'Curah_Hujan': 'Curah Hujan'},
        color_discrete_sequence=['#0000ff']
    )
    fig2.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)",
        xaxis=dict(
            tickformat='%d %b %Y',
            tickangle=-45
        ),
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Statistik curah hujan
    rain_mean = filtered_vis_data['Curah_Hujan'].mean()
    rain_max = filtered_vis_data['Curah_Hujan'].max()
    rain_total = filtered_vis_data['Curah_Hujan'].sum()
    
    st.markdown("<h4 style='text-align: center; color: #0000ff; margin-top: 20px;'>Statistik Curah Hujan</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e3f2fd; border-top: 3px solid #0000ff;">
            <div class="stat-title">Rata-Rata Harian</div>
            <div class="stat-value" style="color: #0000ff;">{rain_mean:.1f} mm</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e3f2fd; border-top: 3px solid #0000ff;">
            <div class="stat-title">Curah Tertinggi</div>
            <div class="stat-value" style="color: #0000ff;">{rain_max:.1f} mm</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e3f2fd; border-top: 3px solid #0000ff;">
            <div class="stat-title">Total Periode</div>
            <div class="stat-value" style="color: #0000ff;">{rain_total:.1f} mm</div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h3 style='text-align: center; color: #43A047;'>Pola Kelembaban Rata-Rata</h3>", unsafe_allow_html=True)
    
    # Membuat grafik
    fig3 = px.line(
        filtered_vis_data,
        x='Tanggal',
        y='Kelembaban_Rata_Rata',
        labels={'Kelembaban_Rata_Rata': 'Kelembaban'},
        color_discrete_sequence=['#008000']
    )
    fig3.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        xaxis_title="Tanggal",
        yaxis_title="Kelembaban (%)",
        margin=dict(t=40, b=40),
        xaxis=dict(
            tickformat='%d %b %Y',
            tickangle=-45
        )
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Statistik kelembaban
    humid_mean = filtered_vis_data['Kelembaban_Rata_Rata'].mean()
    humid_max = filtered_vis_data['Kelembaban_Rata_Rata'].max()
    humid_min = filtered_vis_data['Kelembaban_Rata_Rata'].min()
    
    st.markdown("<h4 style='text-align: center; color: #43A047; margin-top: 20px;'>Statistik Kelembaban</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e8f5e9; border-top: 3px solid #43A047;">
            <div class="stat-title">Rata-Rata</div>
            <div class="stat-value" style="color: #43A047;">{humid_mean:.1f} %</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e8f5e9; border-top: 3px solid #43A047;">
            <div class="stat-title">Tertinggi</div>
            <div class="stat-value" style="color: #43A047;">{humid_max:.1f} %</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #e8f5e9; border-top: 3px solid #43A047;">
            <div class="stat-title">Terendah</div>
            <div class="stat-value" style="color: #43A047;">{humid_min:.1f} %</div>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h3 style='text-align: center; color: #FB8C00;'>Pola Sinar Matahari</h3>", unsafe_allow_html=True)
    
    # Membuat grafik
    fig4 = px.line(
        filtered_vis_data,
        x='Tanggal',
        y='Sinar_Matahari',
        labels={'Sinar_Matahari': 'Sinar Matahari'},
        color_discrete_sequence=['#FB8C00']
    )
    fig4.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(240,240,240,0.8)',
        xaxis_title="Tanggal",
        yaxis_title="Sinar Matahari (Jam)",
        margin=dict(t=40, b=40),
        xaxis=dict(
            tickformat='%d %b %Y',
            tickangle=-45
        )
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Statistik sinar matahari
    sun_mean = filtered_vis_data['Sinar_Matahari'].mean()
    sun_max = filtered_vis_data['Sinar_Matahari'].max()
    sun_min = filtered_vis_data['Sinar_Matahari'].min()
    
    st.markdown("<h4 style='text-align: center; color: #FB8C00; margin-top: 20px;'>Statistik Sinar Matahari</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #fff3e0; border-top: 3px solid #FB8C00;">
            <div class="stat-title">Rata-Rata Harian</div>
            <div class="stat-value" style="color: #FB8C00;">{sun_mean:.1f} Jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #fff3e0; border-top: 3px solid #FB8C00;">
            <div class="stat-title">Durasi Terlama</div>
            <div class="stat-value" style="color: #FB8C00;">{sun_max:.1f} Jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background-color: #fff3e0; border-top: 3px solid #FB8C00;">
            <div class="stat-title">Durasi Terpendek</div>
            <div class="stat-value" style="color: #FB8C00;">{sun_min:.1f} Jam</div>
        </div>
        """, unsafe_allow_html=True)






# 🤖Pembagian Dataset Training dan Testing 🤖

st.markdown("---")
st.markdown(
    "<h2 style='text-align: center;'>🔢 Pembagian Dataset Training dan Testing 🔢</h2>",
    unsafe_allow_html=True
)

# Muat data dari file
file_path = os.path.join(os.path.dirname(__file__), 'Split_Data', 'split_data_kelembapan.pkl')
X_train, X_test, y_train, y_test = joblib.load(file_path)
# Total sampel untuk perhitungan persentase
total_samples = X_train.shape[0] + X_test.shape[0]
training_ratio = X_train.shape[0] / total_samples * 100
testing_ratio = X_test.shape[0] / total_samples * 100

# Definisi CSS untuk kartu dan tampilan visual
st.markdown("""
<style>
    .dataset-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    .training-card {
        border-left: 4px solid #1e90ff;
        background-color: #f0f8ff;
    }
    .testing-card {
        border-left: 4px solid #ff69b4;
        background-color: #fff0f5;
    }
    .summary-card {
        border-left: 4px solid #2e8b57;
        background-color: #f0fff0;
    }
    .feature-title {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 18px;
    }
    .dataset-card p {
        font-size: 16px;
        margin: 5px 0;
    }
    .dataset-card strong {
        font-size: 17px;
    }
    .card-header {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Tata letak kolom
col1, col2, col3 = st.columns(3)

# Kolom 1: Training Set
with col1:
    st.markdown("<div class='card-header'>Training Set</div>", unsafe_allow_html=True)
    
    # Kartu untuk fitur X
    st.markdown(f"""
    <div class="dataset-card training-card">
        <div class="feature-title" style="color: #1e90ff;">Fitur (X)</div>
        <p style="color: #1e90ff;">Jumlah sampel: <strong>{X_train.shape[0]:,}</strong></p>
        <p style="color: #1e90ff;">Jumlah fitur: <strong>{X_train.shape[1]}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Kartu untuk target y
    st.markdown(f"""
    <div class="dataset-card training-card">
        <div class="feature-title" style="color: #1e90ff;">Target (y)</div>
        <p style="color: #1e90ff;">Jumlah sampel: <strong>{y_train.shape[0]:,}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Kolom 2: Testing Set
with col2:
    st.markdown("<div class='card-header'>Testing Set</div>", unsafe_allow_html=True)
    
    # Kartu untuk fitur X
    st.markdown(f"""
    <div class="dataset-card testing-card">
        <div class="feature-title" style="color: #ff69b4;">Fitur (X)</div>
        <p style="color: #ff69b4;">Jumlah sampel: <strong>{X_test.shape[0]:,}</strong></p>
        <p style="color: #ff69b4;">Jumlah fitur: <strong>{X_test.shape[1]}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Kartu untuk target y
    st.markdown(f"""
    <div class="dataset-card testing-card">
        <div class="feature-title" style="color: #ff69b4;">Target (y)</div>
        <p style="color: #ff69b4;">Jumlah sampel: <strong>{y_test.shape[0]:,}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Kolom 3: Ringkasan
with col3:
    st.markdown("<div class='card-header'>Ringkasan Pembagian Data</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dataset-card summary-card">
        <p style="color: #2e8b57;">Total dataset: <strong>{total_samples:,}</strong> sampel</p>
        <p style="color: #2e8b57;">Rasio training: <strong>{training_ratio:.1f}%</strong></p>
        <p style="color: #2e8b57;">Rasio testing: <strong>{testing_ratio:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)





# 🤖🤖Evaluasi Model🤖🤖

# Load model .joblib
model_rf = joblib.load("./Model/model_kelembapan_rf.joblib")
model_gb = joblib.load("./Model/model_kelembapan_gb.joblib")
y_pred_xgb = joblib.load("./Model/model_kelembapan_xgb.joblib")

st.markdown("---")
st.markdown("<h1 style='text-align: center; padding-top: 20px; padding-bottom: 40px;'>Evaluasi Model Prediksi Kelembapan</h1>", unsafe_allow_html=True)

tab_rf, tab_gb, tab_xgb, tab_compare = st.tabs(["Random Forest", "Gradient Boosting", "XGBoost", "Perbandingan Semua Model"])

#-------- Random Forest Tab --------
with tab_rf:
    st.markdown("<h2 style='text-align: center;'>Model Random Forest</h2>", unsafe_allow_html=True)
    
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

    # Visualisasi metrik
    st.markdown("<h3 style='text-align: center;'>📈 Visualisasi Metrik Evaluasi </h3>", unsafe_allow_html=True)
    metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
    values = [mae_rf, mse_rf, rmse_rf, r2_rf, mape_rf]
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
            
    metrics_fig = px.bar(
                x=metrics,
                y=values,
                color=metrics,
                color_discrete_sequence=colors,
                text=[f'{v:.4f}' if metric != 'MAPE' else f'{v:.2f}%' for metric, v in zip(metrics, values)],
                labels={'x': 'Metrik', 'y': 'Nilai'},
            )
    metrics_fig.update_layout(
                hovermode="x",
                plot_bgcolor='rgba(240,240,240,0.8)',
                margin=dict(t=40, b=80),
                xaxis_title='',
                yaxis_title="Nilai Metrik",
                showlegend=False
            )
    st.plotly_chart(metrics_fig, use_container_width=True)

    # Kartu metrik dalam kolom
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #F44336; background-color: #FFEBEE; padding: 5px; border-radius: 8px;">
            <div class="metric-title" style='text-align: center; color: #F44336'>MAE</div>
            <div class="metric-value" style="color: #F44336; text-align: center; font-weight: bold; font-size: 24px">{mae_rf:.4f}</div>
            <div class="metric-unit" style='text-align: center; color: #F44336'>Rata-rata selisih absolut</div>
            </div>
            """, unsafe_allow_html=True
        )
            
    with col2:
        st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;border-left: 4px solid #2196F3; background-color: #E3F2FD; padding: 5px; border-radius: 8px;">
                    <div class="metric-title" style="color: #2196F3; text-align: center;">MSE</div>
                    <div class="metric-value" style="color: #2196F3; text-align: center; font-weight: bold; font-size: 24px">{mse_rf:.4f}</div>
                    <div class="metric-unit" style="color: #2196F3; text-align: center;">Rata-rata selisih kuadrat </div>
                </div>
                """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #4CAF50; background-color: #E8F5E9; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #4CAF50;">RMSE</div>
                <div class="metric-value" style="color: #4CAF50; text-align: center; font-weight: bold; font-size: 24px">{rmse_rf:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #4CAF50;">Akar kuadrat dari MSE</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #FF9800; background-color: #FFF3E0; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #FF9800;">R² Score</div>
                <div class="metric-value" style="color: #FF9800; text-align: center; font-weight: bold; font-size: 24px">{r2_rf:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #FF9800;">Kecocokan model</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #9C27B0; background-color: #F3E5F5; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #9C27B0;">MAPE</div>
                <div class="metric-value" style="color: #9C27B0; text-align: center; font-weight: bold; font-size: 24px">{mape_rf:.2f}%</div>
                <div class="metric-unit" style="text-align: center; color: #9C27B0;">Persentase kesalahan</div>
            </div>
        """, unsafe_allow_html=True)

    # Perbandingan Nilai Aktual dan Prediksi
    st.markdown("<h3 style='text-align: center; padding-top: 50px'>🔮 Perbandingan Nilai Aktual dan Prediksi</h3>", unsafe_allow_html=True)

    # Buat figure Plotly
    actual_vs_pred_fig = go.Figure()

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_test)),
        y=y_test.values,
        name='Aktual',
        line=dict(color= '#008000', width=2),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_rf)),
        y=y_pred_rf,
        name='Prediksi',
        line=dict(color='#4321ce ', width=2, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.update_layout(
        height=600,
        xaxis_title='Index',
        yaxis_title='Kelembapan Rata Rata (%)',
        title={
            'text': 'Perbandingan Prediksi dan Aktual Kelembapan Rata Rata (Random Forest)',
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
    # Tampilkan plot di Streamlit
    st.plotly_chart(actual_vs_pred_fig, use_container_width=True)

#-------- Gradient Boosting Tab --------
with tab_gb:
    st.markdown("<h2 style='text-align: center;'>Model Gradient Boosting</h2>", unsafe_allow_html=True)
    
    model_gb.fit(X_train, y_train)
    y_pred_gb = model_gb.predict(X_test)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mse_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    mape_gb = np.mean(np.abs((y_test - y_pred_gb) / y_test)) * 100

    # Visualisasi metrik
    st.markdown("<h3 style='text-align: center;'>📈 Visualisasi Metrik Evaluasi </h3>", unsafe_allow_html=True)
    metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
    values = [mae_gb, mse_gb, rmse_gb, r2_gb, mape_gb]
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
            
    metrics_fig = px.bar(
                x=metrics,
                y=values,
                color=metrics,
                color_discrete_sequence=colors,
                text=[f'{v:.4f}' if metric != 'MAPE' else f'{v:.2f}%' for metric, v in zip(metrics, values)],
                labels={'x': 'Metrik', 'y': 'Nilai'},
            )
    metrics_fig.update_layout(
                hovermode="x",
                plot_bgcolor='rgba(240,240,240,0.8)',
                margin=dict(t=40, b=80),
                xaxis_title='',
                yaxis_title="Nilai Metrik",
                showlegend=False
            )
    st.plotly_chart(metrics_fig, use_container_width=True)

    # Metric cards in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #F44336; background-color: #FFEBEE; padding: 5px; border-radius: 8px;">
            <div class="metric-title" style='text-align: center; color: #F44336'>MAE</div>
            <div class="metric-value" style="color: #F44336; text-align: center; font-weight: bold; font-size: 24px">{mae_gb:.4f}</div>
            <div class="metric-unit" style='text-align: center; color: #F44336'>Rata-rata selisih absolut</div>
            </div>
            """, unsafe_allow_html=True
        )
            
    with col2:
        st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;border-left: 4px solid #2196F3; background-color: #E3F2FD; padding: 5px; border-radius: 8px;">
                    <div class="metric-title" style="color: #2196F3; text-align: center;">MSE</div>
                    <div class="metric-value" style="color: #2196F3; text-align: center; font-weight: bold; font-size: 24px">{mse_gb:.4f}</div>
                    <div class="metric-unit" style="color: #2196F3; text-align: center;">Rata-rata selisih kuadrat </div>
                </div>
                """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #4CAF50; background-color: #E8F5E9; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #4CAF50;">RMSE</div>
                <div class="metric-value" style="color: #4CAF50; text-align: center; font-weight: bold; font-size: 24px">{rmse_gb:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #4CAF50;">Akar kuadrat dari MSE</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #FF9800; background-color: #FFF3E0; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #FF9800;">R² Score</div>
                <div class="metric-value" style="color: #FF9800; text-align: center; font-weight: bold; font-size: 24px">{r2_gb:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #FF9800;">Kecocokan model</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #9C27B0; background-color: #F3E5F5; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #9C27B0;">MAPE</div>
                <div class="metric-value" style="color: #9C27B0; text-align: center; font-weight: bold; font-size: 24px">{mape_gb:.2f}%</div>
                <div class="metric-unit" style="text-align: center; color: #9C27B0;">Persentase kesalahan</div>
            </div>
        """, unsafe_allow_html=True)

    # Perbandingan Nilai Aktual dan Prediksi
    st.markdown("<h3 style='text-align: center; padding-top: 50px'>🔮 Perbandingan Nilai Aktual dan Prediksi</h3>", unsafe_allow_html=True)

    # Buat figure Plotly
    actual_vs_pred_fig = go.Figure()

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_test)),
        y=y_test.values,
        name='Aktual',
        line=dict(color= '#008000', width=2),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_gb)),
        y=y_pred_gb,
        name='Prediksi',
        line=dict(color='#f7592f', width=2, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.update_layout(
        height=600,
        xaxis_title='Index',
        yaxis_title='Kelembapan Rata Rata (%)',
        title={
            'text': 'Perbandingan Prediksi dan Aktual Kelembapan Rata Rata (Gradient Boosting)',
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
    # Tampilkan plot di Streamlit
    st.plotly_chart(actual_vs_pred_fig, use_container_width=True)

#-------- XGBoost Tab --------
with tab_xgb:
    st.markdown("<h2 style='text-align: center;'>Model XGBoost</h2>", unsafe_allow_html=True)
    
    # model_xgb.fit(X_train, y_train)
    # y_pred_xgb = model_xgb.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

    # Visualisasi metrik
    st.markdown("<h3 style='text-align: center;'>📈 Visualisasi Metrik Evaluasi </h3>", unsafe_allow_html=True)
    metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
    values = [mae_xgb, mse_xgb, rmse_xgb, r2_xgb, mape_xgb]
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
            
    metrics_fig = px.bar(
                x=metrics,
                y=values,
                color=metrics,
                color_discrete_sequence=colors,
                text=[f'{v:.4f}' if metric != 'MAPE' else f'{v:.2f}%' for metric, v in zip(metrics, values)],
                labels={'x': 'Metrik', 'y': 'Nilai'},
            )
    metrics_fig.update_layout(
                hovermode="x",
                plot_bgcolor='rgba(240,240,240,0.8)',
                margin=dict(t=40, b=80),
                xaxis_title='',
                yaxis_title="Nilai Metrik",
                showlegend=False
            )
    st.plotly_chart(metrics_fig, use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #F44336; background-color: #FFEBEE; padding: 5px; border-radius: 8px;">
            <div class="metric-title" style='text-align: center; color: #F44336'>MAE</div>
            <div class="metric-value" style="color: #F44336; text-align: center; font-weight: bold; font-size: 24px">{mae_xgb:.4f}</div>
            <div class="metric-unit" style='text-align: center; color: #F44336'>Rata-rata selisih absolut</div>
            </div>
            """, unsafe_allow_html=True
        )
            
    with col2:
        st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;border-left: 4px solid #2196F3; background-color: #E3F2FD; padding: 5px; border-radius: 8px;">
                    <div class="metric-title" style="color: #2196F3; text-align: center;">MSE</div>
                    <div class="metric-value" style="color: #2196F3; text-align: center; font-weight: bold; font-size: 24px">{mse_xgb:.4f}</div>
                    <div class="metric-unit" style="color: #2196F3; text-align: center;">Rata-rata selisih kuadrat </div>
                </div>
                """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #4CAF50; background-color: #E8F5E9; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #4CAF50;">RMSE</div>
                <div class="metric-value" style="color: #4CAF50; text-align: center; font-weight: bold; font-size: 24px">{rmse_xgb:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #4CAF50;">Akar kuadrat dari MSE</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #FF9800; background-color: #FFF3E0; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #FF9800;">R² Score</div>
                <div class="metric-value" style="color: #FF9800; text-align: center; font-weight: bold; font-size: 24px">{r2_xgb:.4f}</div>
                <div class="metric-unit" style="text-align: center; color: #FF9800;">Kecocokan model</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-card" style="min-height: 120px;border-left: 4px solid #9C27B0; background-color: #F3E5F5; padding: 5px; border-radius: 8px;">
                <div class="metric-title" style="text-align: center; color: #9C27B0;">MAPE</div>
                <div class="metric-value" style="color: #9C27B0; text-align: center; font-weight: bold; font-size: 24px">{mape_xgb:.2f}%</div>
                <div class="metric-unit" style="text-align: center; color: #9C27B0;">Persentase kesalahan</div>
            </div>
        """, unsafe_allow_html=True)

    # Perbandingan Nilai Aktual dan Prediksi
    st.markdown("<h3 style='text-align: center; padding-top: 50px'>🔮 Perbandingan Nilai Aktual dan Prediksi</h3>", unsafe_allow_html=True)

    # Buat figure Plotly
    actual_vs_pred_fig = go.Figure()

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_test)),
        y=y_test.values,
        name='Aktual',
        line=dict(color= '#008000', width=2),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_xgb)),
        y=y_pred_xgb,
        name='Prediksi',
        line=dict(color='#1c39f4', width=2, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai: %{y:.2f}%'
    ))

    actual_vs_pred_fig.update_layout(
        height=600,
        xaxis_title='Index',
        yaxis_title='Kelembapan Rata Rata (%)',
        title={
            'text': 'Perbandingan Prediksi dan Aktual Kelembapan Rata Rata (XGBoost)',
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
    # Tampilkan plot di Streamlit
    st.plotly_chart(actual_vs_pred_fig, use_container_width=True)

#-------- Comparison Tab --------
with tab_compare:
    st.markdown("<h2 style='text-align: center;'>Perbandingan Semua Kinerja Model (Kelembapan Rata-Rata)</h2>", unsafe_allow_html=True)

    metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
    rf_scores = [mae_rf, mse_rf, rmse_rf, r2_rf, mape_rf]
    gb_scores = [mae_gb, mse_gb, rmse_gb, r2_gb, mape_gb]
    xgb_scores = [mae_xgb, mse_xgb, rmse_xgb, r2_xgb, mape_xgb]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=rf_scores,
        name='Random Forest',
        marker=dict(color='#3498db'),
        text=[f'{v:.3f}' for v in rf_scores],
        textposition='auto',
        hovertemplate='<b>Random Forest</b><br>%{x}: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=gb_scores,
        name='Gradient Boosting',
        marker=dict(color='#2ecc71'),
        text=[f'{v:.3f}' for v in gb_scores],
        textposition='auto',
        hovertemplate='<b>Gradient Boosting</b><br>%{x}: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=xgb_scores,
        name='XGBoost',
        marker=dict(color='#e74c3c'),
        text=[f'{v:.3f}' for v in xgb_scores],
        textposition='auto',
        hovertemplate='<b>XGBoost</b><br>%{x}: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Perbandingan Kinerja Model (Kelembapan Rata-Rata)',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Metrik Evaluasi",
        yaxis_title="Nilai",
        legend_title="Model",
        barmode='group',
        bargap=0.25,
        bargroupgap=0.1,
        height=550,
        font=dict(size=13),
        
        template='none',
        plot_bgcolor='white',
        paper_bgcolor='white',

        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; padding-top: 20px'>📊 Tabel Perbandingan Metrik</h3>", unsafe_allow_html=True)
    
    comparison_df = pd.DataFrame({
        'Metrik': metrics,
        'Random Forest': [f"{score:.4f}" if i != 4 else f"{score:.2f}%" for i, score in enumerate(rf_scores)],
        'Gradient Boosting': [f"{score:.4f}" if i != 4 else f"{score:.2f}%" for i, score in enumerate(gb_scores)],
        'XGBoost': [f"{score:.4f}" if i != 4 else f"{score:.2f}%" for i, score in enumerate(xgb_scores)]
    })
    
    def highlight_best(row):
        if row.name == 3:
            best = row.iloc[1:].astype(float).idxmax()
            styles = ['' if col == best else '' for col in row.index]
            return styles
        else:
            values = [float(val.replace('%', '')) if '%' in str(val) else float(val) for val in row.iloc[1:]]
            best_idx = values.index(min(values)) + 1 
            styles = ['' if i == best_idx else '' for i in range(len(row))]
            return styles
    
    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; padding-top: 30px'>🔄 Perbandingan Prediksi Semua Model</h3>", unsafe_allow_html=True)
    
    all_models_fig = go.Figure()
    
    all_models_fig.add_trace(go.Scatter(
        x=np.arange(len(y_test)),
        y=y_test.values,
        name='Aktual',
        line=dict(color='#008000', width=2.5),
        mode='lines',
        hovertemplate='Index: %{x}<br>Nilai Aktual: %{y:.2f}%'
    ))
    
    all_models_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_rf)),
        y=y_pred_rf,
        name='Random Forest',
        line=dict(color='#4321ce', width=1.5, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Prediksi RF: %{y:.2f}%'
    ))
    
    all_models_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_gb)),
        y=y_pred_gb,
        name='Gradient Boosting',
        line=dict(color='#f7592f', width=1.5, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Prediksi GB: %{y:.2f}%'
    ))
    
    all_models_fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred_xgb)),
        y=y_pred_xgb,
        name='XGBoost',
        line=dict(color='#1c39f4', width=1.5, dash='solid'),
        mode='lines',
        hovertemplate='Index: %{x}<br>Prediksi XGB: %{y:.2f}%'
    ))
    
    all_models_fig.update_layout(
        height=600,
        xaxis_title='Index',
        yaxis_title='Kelembapan Rata Rata (%)',
        title={
            'text': 'Perbandingan Prediksi Semua Model dengan Nilai Aktual',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
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
    
    st.plotly_chart(all_models_fig, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; padding-top: 30px'>📉 Analisis Kesalahan Model</h3>", unsafe_allow_html=True)
    
    errors_rf = y_test.values - y_pred_rf
    errors_gb = y_test.values - y_pred_gb
    errors_xgb = y_test.values - y_pred_xgb
    
    error_fig = go.Figure()
    
    error_fig.add_trace(go.Histogram(
        x=errors_rf,
        name='Random Forest',
        opacity=0.7,
        marker=dict(color='#4321ce'),
        nbinsx=30,
        hovertemplate='Error: %{x:.2f}<br>Frekuensi: %{y}<extra>Random Forest</extra>'
    ))
    
    error_fig.add_trace(go.Histogram(
        x=errors_gb,
        name='Gradient Boosting',
        opacity=0.7,
        marker=dict(color='#f7592f'),
        nbinsx=30,
        hovertemplate='Error: %{x:.2f}<br>Frekuensi: %{y}<extra>Gradient Boosting</extra>'
    ))
    
    error_fig.add_trace(go.Histogram(
        x=errors_xgb,
        name='XGBoost',
        opacity=0.7,
        marker=dict(color='#1c39f4'),
        nbinsx=30,
        hovertemplate='Error: %{x:.2f}<br>Frekuensi: %{y}<extra>XGBoost</extra>'
    ))
    
    error_fig.update_layout(
        barmode='overlay',
        title={
            'text': 'Distribusi Kesalahan Prediksi Model',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Kesalahan Prediksi (Aktual - Prediksi)",
        yaxis_title="Frekuensi",
        legend_title="Model",
        height=500,
        plot_bgcolor='white',
        bargap=0.1,
        bargroupgap=0.2
    )
    
    st.plotly_chart(error_fig, use_container_width=True)
    
    error_stats = pd.DataFrame({
        'Statistik': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
        'Random Forest': [
            f"{errors_rf.min():.4f}",
            f"{errors_rf.max():.4f}",
            f"{errors_rf.mean():.4f}",
            f"{np.median(errors_rf):.4f}",
            f"{errors_rf.std():.4f}"
        ],
        'Gradient Boosting': [
            f"{errors_gb.min():.4f}",
            f"{errors_gb.max():.4f}",
            f"{errors_gb.mean():.4f}",
            f"{np.median(errors_gb):.4f}",
            f"{errors_gb.std():.4f}"
        ],
        'XGBoost': [
            f"{errors_xgb.min():.4f}",
            f"{errors_xgb.max():.4f}",
            f"{errors_xgb.mean():.4f}",
            f"{np.median(errors_xgb):.4f}",
            f"{errors_xgb.std():.4f}"
        ]
    })
    
    st.markdown("<h4 style='text-align: center; padding-top: 10px'>Statistik Kesalahan Prediksi</h4>", unsafe_allow_html=True)
    st.dataframe(error_stats, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; padding-top: 30px'>🏆 Kesimpulan dan Rekomendasi</h3>", unsafe_allow_html=True)
    
    r2_scores = [r2_rf, r2_gb, r2_xgb]
    rmse_scores = [rmse_rf, rmse_gb, rmse_xgb]
    model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost']
    
    best_r2_idx = r2_scores.index(max(r2_scores))
    best_rmse_idx = rmse_scores.index(min(rmse_scores))
    
    if best_r2_idx == best_rmse_idx:
        best_model = model_names[best_r2_idx]
        best_r2 = r2_scores[best_r2_idx]
        best_rmse = rmse_scores[best_rmse_idx]
        
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; text-align: center">
            <h4 style="color: #2196F3; margin-top: 0;">Model Terbaik: {best_model}</h4>
            <p style="color: #2196F3; margin-top: 0;">Berdasarkan evaluasi metrik, model <b style="color: #2196F3; font-weight: bold">{best_model}</b> menunjukkan performa terbaik dengan nilai:</p>
            <ul style="color: #2196F3; margin-top: 0;">
                <li >R² Score: <b>{best_r2:.4f}</b> (semakin tinggi semakin baik)</li>
                <li>RMSE: <b>{best_rmse:.4f}</b> (semakin rendah semakin baik)</li>
            </ul>
            <p style="color: #2196F3; margin-top: 0;">Model ini memiliki keseimbangan terbaik dalam akurasi prediksi kelembapan dan konsistensi performa.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;">
            <h4 style="color: #2196F3; margin-top: 0;">Rekomendasi Model</h4>
            <p>Berdasarkan evaluasi metrik yang berbeda:</p>
            <ul>
                <li><b>{model_names[best_r2_idx]}</b> memberikan R² Score terbaik: <b>{r2_scores[best_r2_idx]:.4f}</b></li>
                <li><b>{model_names[best_rmse_idx]}</b> memberikan RMSE terendah: <b>{rmse_scores[best_rmse_idx]:.4f}</b></li>
            </ul>
            <p>Pilihan model tergantung pada prioritas:</p>
            <ul>
                <li>Jika prioritas adalah kecocokan model secara keseluruhan, pilih <b>{model_names[best_r2_idx]}</b></li>
                <li>Jika prioritas adalah akurasi prediksi dengan kesalahan minimum, pilih <b>{model_names[best_rmse_idx]}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



# >🤖🤖Prediksi Kelembapan Udara🤖🤖
st.markdown("---")
st.markdown("<h1 style='text-align: center; padding-buttom: 80px;'>💧Prediksi Kelembapan Udara💧</h1>", unsafe_allow_html=True)

# Fungsi load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('./Model/model_kelembapan_rf.joblib')
    except FileNotFoundError:
        st.error("Model file tidak ditemukan. Pastikan file 'model_kelembapan_rf.joblib' ada di direktori yang sama.")
        return None

# Fungsi untuk memprediksi kelembaban di masa mendatang
def prediksi_kelembapan_masa_depan(data_terakhir, hari_untuk_diprediksi, model):
    """
    Memprediksi kelembapan rata-rata untuk beberapa hari ke depan.

    Args:
        data_terakhir: DataFrame dengan data terbaru yang digunakan sebagai dasar prediksi
        hari_untuk_diprediksi: Jumlah hari yang akan diprediksi
        model: Model Random Forest yang telah dilatih
    Returns:
        DataFrame berisi tanggal dan prediksi kelembapan
    """
    # Inisialisasi dataframe untuk menyimpan hasil prediksi
    hasil_prediksi = pd.DataFrame(columns=['TANGGAL', 'Kelembapan_Rata-Rata'])
    
    # Mendapatkan tanggal terakhir dari data
    tanggal_terakhir = data_terakhir['TANGGAL'].iloc[0]
    
    # Menyiapkan data untuk prediksi
    data_baru = data_terakhir.copy()
    
    # Fitur-fitur yang digunakan untuk prediksi
    fitur_dasar = [
        'Suhu_Rata-Rata',
        'Curah_Hujan',
        'Sinar_Matahari',
    ]
    fitur_lag = [
        'Suhu_Rata-Rata_1HariSebelum', 'Kelembapan_1HariSebelum',
        'Hujan_1HariSebelum', 'Matahari_1HariSebelum'
    ]
    fitur_rolling = [
        'Suhu_Rata-Rata_Rolling3Hari', 'Kelembapan_Rolling3Hari',
        'Hujan_Rolling3Hari', 'Matahari_Rolling3Hari'
    ]
    fitur_waktu = ['Hari', 'Bulan', 'Tahun']
    
    # Semua fitur yang digunakan
    fitur = fitur_dasar + fitur_lag + fitur_rolling + fitur_waktu
    
    # Iterasi untuk memprediksi hari demi hari
    for i in range(hari_untuk_diprediksi):
        # Tanggal untuk prediksi
        tanggal_prediksi = tanggal_terakhir + timedelta(days=i+1)
        
        # Update fitur waktu untuk tanggal prediksi
        data_baru['TANGGAL'] = tanggal_prediksi
        data_baru['Hari'] = tanggal_prediksi.dayofweek
        data_baru['Bulan'] = tanggal_prediksi.month
        data_baru['Tahun'] = tanggal_prediksi.year
        
        # Prediksi kelembapan untuk hari ini
        X_predict = data_baru[fitur]
        kelembapan_prediction = model.predict(X_predict)[0]
        
        # Menambahkan hasil prediksi ke DataFrame hasil_prediksi
        hasil_prediksi.loc[i] = [tanggal_prediksi, kelembapan_prediction]
        
        # Update data untuk prediksi hari berikutnya
        data_baru['Kelembapan_1HariSebelum'] = kelembapan_prediction
        data_baru['Suhu_Rata-Rata_1HariSebelum'] = data_baru['Suhu_Rata-Rata']
        data_baru['Hujan_1HariSebelum'] = data_baru['Curah_Hujan']
        data_baru['Matahari_1HariSebelum'] = data_baru['Sinar_Matahari']
        
        # Update rolling average setelah 3 hari
        if i >= 2:
            prev_3_kelembapan = hasil_prediksi.loc[i-2:i, 'Kelembapan_Rata-Rata'].values
            data_baru['Kelembapan_Rolling3Hari'] = np.mean(prev_3_kelembapan)
            
            # Update rolling untuk fitur lain jika ada data historis
            # Untuk demonstrasi, kita biarkan nilai Rolling3Hari untuk fitur lain tetap sama
    
    return hasil_prediksi

# Load model
model_rf = load_model()

# Header untuk input parameters menggunakan expander
st.markdown('<h2 style="text-align: center; padding: 20px;">🔢 Parameter Prediksi 🔢</h2>', unsafe_allow_html=True)

# Menggunakan expander untuk form input
with st.expander("🔽 Input Data Kondisi Terkini 🔽", expanded=False):
    with st.form("input_form"):
        st.subheader("Data Kondisi Terkini")
        
        # Tanggal
        tanggal = st.date_input(
            "Tanggal Data Terkini",
            datetime.now().date()
        )
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Fitur dasar untuk hari ini
            suhu_rata_rata = st.number_input("Suhu Rata-Rata (°C)", value=28.5, min_value=0.0, max_value=50.0, step=0.1)
            curah_hujan = st.number_input("Curah Hujan (mm)", value=0.5, min_value=0.0, max_value=500.0, step=0.1)
            sinar_matahari = st.number_input("Sinar Matahari (jam)", value=7.5, min_value=0.0, max_value=12.0, step=0.1)
        
        with col2:
            # Fitur lag (1 hari sebelumnya)
            suhu_sebelum = st.number_input("Suhu 1 Hari Sebelum (°C)", value=28.2, min_value=0.0, max_value=50.0, step=0.1)
            kelembapan_sebelum = st.number_input("Kelembapan 1 Hari Sebelum (%)", value=83.0, min_value=0.0, max_value=100.0, step=0.1)
            hujan_sebelum = st.number_input("Curah Hujan 1 Hari Sebelum (mm)", value=1.0, min_value=0.0, max_value=500.0, step=0.1)
            matahari_sebelum = st.number_input("Sinar Matahari 1 Hari Sebelum (jam)", value=6.5, min_value=0.0, max_value=12.0, step=0.1)
        
        st.markdown("---")
        
        # Fitur rolling (rata-rata 3 hari)
        st.subheader("Data Rata-rata 3 Hari Terakhir")
        
        col3, col4 = st.columns(2)
        
        with col3:
            suhu_rolling = st.number_input("Suhu Rolling 3 Hari (°C)", value=28.3, min_value=0.0, max_value=50.0, step=0.1)
            kelembapan_rolling = st.number_input("Kelembapan Rolling 3 Hari (%)", value=82.5, min_value=0.0, max_value=100.0, step=0.1)
        
        with col4:
            hujan_rolling = st.number_input("Curah Hujan Rolling 3 Hari (mm)", value=0.8, min_value=0.0, max_value=500.0, step=0.1)
            matahari_rolling = st.number_input("Sinar Matahari Rolling 3 Hari (jam)", value=7.0, min_value=0.0, max_value=12.0, step=0.1)
        
        st.markdown("---")
        
        # Jumlah hari untuk diprediksi
        st.subheader("Parameter Prediksi")
        hari_prediksi = st.slider("Jumlah Hari untuk Diprediksi", min_value=7, max_value=90, value=30, step=1)
        
        # Submit button
        submitted = st.form_submit_button("Jalankan Prediksi")

# Main content
if model_rf is not None:
    if submitted:
        with st.spinner("Memproses prediksi..."):
            # Membuat DataFrame dari input
            data_terakhir = pd.DataFrame({
                'TANGGAL': [pd.Timestamp(tanggal)],
                'Suhu_Rata-Rata': [suhu_rata_rata],
                'Curah_Hujan': [curah_hujan],
                'Sinar_Matahari': [sinar_matahari],
                'Hari': [pd.Timestamp(tanggal).dayofweek],
                'Bulan': [pd.Timestamp(tanggal).month],
                'Tahun': [pd.Timestamp(tanggal).year],
                'Suhu_Rata-Rata_1HariSebelum': [suhu_sebelum],
                'Kelembapan_1HariSebelum': [kelembapan_sebelum],
                'Hujan_1HariSebelum': [hujan_sebelum],
                'Matahari_1HariSebelum': [matahari_sebelum],
                'Suhu_Rata-Rata_Rolling3Hari': [suhu_rolling],
                'Kelembapan_Rolling3Hari': [kelembapan_rolling],
                'Hujan_Rolling3Hari': [hujan_rolling],
                'Matahari_Rolling3Hari': [matahari_rolling]
            })
            
            # Run prediction
            prediksi_masa_depan = prediksi_kelembapan_masa_depan(data_terakhir, hari_prediksi, model_rf)
            
            # Store the prediction in session state for reuse
            st.session_state.prediksi = prediksi_masa_depan
            st.session_state.has_prediction = True
    
    # Menampilkan hasil jika prediksi ada (baik dari kiriman saat ini atau sebelumnya)
    if 'has_prediction' in st.session_state and st.session_state.has_prediction:
        prediksi_masa_depan = st.session_state.prediksi
        
        # Buat tab untuk tampilan yang berbeda
        tab1, tab2, tab3 = st.tabs(["📈 Grafik Prediksi", "📊 Statistik", "📋 Data Lengkap"])
        
        with tab1:
            st.subheader("Prediksi Kelembapan Udara untuk {} Hari ke Depan".format(len(prediksi_masa_depan)))
            
            # Create Plotly figure
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=prediksi_masa_depan['TANGGAL'],
                    y=prediksi_masa_depan['Kelembapan_Rata-Rata'],
                    mode='lines+markers',
                    name='Kelembapan (%)',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=6, color='darkblue')
                )
            )
            
            fig.update_layout(
                title={
                    'text': f"Prediksi Kelembapan Udara ({prediksi_masa_depan['TANGGAL'].min().strftime('%d %b %Y')} - {prediksi_masa_depan['TANGGAL'].max().strftime('%d %b %Y')})",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Tanggal",
                yaxis_title="Kelembapan Rata-Rata (%)",
                hovermode="x unified",
                height=500,
                xaxis=dict(
                    tickformat="%d %b %Y"
                ),
                yaxis=dict(
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                plot_bgcolor='rgb(255, 255, 255)',
                margin=dict(l=40, r=40, t=80, b=40),
            )
            
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7D", step="day", stepmode="backward"),
                            dict(count=14, label="14D", step="day", stepmode="backward"),
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Statistik Prediksi Kelembapan")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Rata-rata", 
                    f"{prediksi_masa_depan['Kelembapan_Rata-Rata'].mean():.2f}%"
                )
            
            with col2:
                st.metric(
                    "Minimum", 
                    f"{prediksi_masa_depan['Kelembapan_Rata-Rata'].min():.2f}%"
                )
            
            with col3:
                st.metric(
                    "Maksimum", 
                    f"{prediksi_masa_depan['Kelembapan_Rata-Rata'].max():.2f}%"
                )
            
            with col4:
                st.metric(
                    "Standar Deviasi", 
                    f"{prediksi_masa_depan['Kelembapan_Rata-Rata'].std():.2f}%"
                )
            
            st.subheader("Distribusi Kelembapan")
            hist_fig = px.histogram(
                prediksi_masa_depan, 
                x='Kelembapan_Rata-Rata',
                nbins=20,
                labels={'Kelembapan_Rata-Rata': 'Kelembapan (%)'},
                title='Distribusi Prediksi Kelembapan',
                color_discrete_sequence=['royalblue']
            )
            hist_fig.update_layout(
                xaxis_title="Kelembapan (%)",
                yaxis_title="Frekuensi",
                bargap=0.1,
                height=400,
                plot_bgcolor='rgb(255, 255, 255)'
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            # Menampilkan rangkuman berdasarkan bulan
            if len(prediksi_masa_depan) > 14:  # Hanya jika ada cukup data
                st.subheader("Rangkuman Berdasarkan Bulan")
                
                # Menambahkan nama bulan ke dataframe
                prediksi_masa_depan['Bulan_Nama'] = prediksi_masa_depan['TANGGAL'].dt.strftime('%B %Y')
                
                # Menghitung rata-rata per bulan
                monthly_avg = prediksi_masa_depan.groupby('Bulan_Nama')['Kelembapan_Rata-Rata'].agg(['mean', 'min', 'max']).reset_index()
                monthly_avg.columns = ['Bulan', 'Rata-rata', 'Minimum', 'Maksimum']
                
                # Menampilkan sebagai bar chart
                monthly_fig = px.bar(
                    monthly_avg,
                    x='Bulan',
                    y='Rata-rata',
                    error_y=monthly_avg['Maksimum'] - monthly_avg['Rata-rata'],
                    error_y_minus=monthly_avg['Rata-rata'] - monthly_avg['Minimum'],
                    title='Rata-rata Kelembapan Bulanan',
                    color_discrete_sequence=['royalblue']
                )
                monthly_fig.update_layout(
                    xaxis_title="Bulan",
                    yaxis_title="Kelembapan Rata-rata (%)",
                    height=400,
                    plot_bgcolor='rgb(255, 255, 255)'
                )
                st.plotly_chart(monthly_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Data Lengkap Prediksi Kelembapan")
            
            # Format tanggal untuk tampilan
            prediksi_display = prediksi_masa_depan.copy()
            prediksi_display['TANGGAL'] = prediksi_display['TANGGAL'].dt.strftime('%Y-%m-%d')
            prediksi_display['Kelembapan_Rata-Rata'] = prediksi_display['Kelembapan_Rata-Rata'].round(2)
            prediksi_display.rename(columns={'TANGGAL': 'Tanggal', 'Kelembapan_Rata-Rata': 'Kelembapan Rata-Rata (%)'}, inplace=True)
            
            st.dataframe(prediksi_display, use_container_width=True)
            
            csv = prediksi_display.to_csv(index=False)
            tanggal_awal = prediksi_masa_depan['TANGGAL'].min().strftime('%Y%m%d')
            tanggal_akhir = prediksi_masa_depan['TANGGAL'].max().strftime('%Y%m%d')
            
            st.download_button(
                label="📥 Download Data Prediksi (CSV)",
                data=csv,
                file_name=f"prediksi_kelembapan_{tanggal_awal}_s.d_{tanggal_akhir}.csv",
                mime="text/csv",
            )
    else:
        st.info("☝️ Masukkan parameter input di panel sebelah atas, lalu klik 'Jalankan Prediksi' untuk menghasilkan prediksi kelembapan udara.")
        
        st.subheader("Contoh Visualisasi")
        
        tanggal_contoh = pd.date_range(start=datetime.now().date(), periods=30)
        kelembapan_contoh = np.random.normal(loc=82, scale=2, size=30).round(2)
        df_contoh = pd.DataFrame({
            'TANGGAL': tanggal_contoh,
            'Kelembapan_Rata-Rata': kelembapan_contoh
        })
        
        fig_contoh = go.Figure()
        fig_contoh.add_trace(
            go.Scatter(
                x=df_contoh['TANGGAL'],
                y=df_contoh['Kelembapan_Rata-Rata'],
                mode='lines+markers',
                name='Kelembapan (%)',
                line=dict(color='lightgrey', width=2, dash='dash'),
                marker=dict(size=5, color='darkgrey')
            )
        )
        fig_contoh.update_layout(
            title="Contoh Visualisasi Prediksi Kelembapan (Data Simulasi)",
            xaxis_title="Tanggal",
            yaxis_title="Kelembapan Rata-Rata (%)",
            height=400,
            plot_bgcolor='rgb(255, 255, 255)',
        )
        
        st.plotly_chart(fig_contoh, use_container_width=True)
        
        st.caption("Catatan: Visualisasi di atas menggunakan data simulasi. Masukkan parameter input untuk melihat prediksi yang sebenarnya.")

# Footer
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>📚 Informasi Model</h3>", unsafe_allow_html=True)
st.markdown("""
Model prediksi kelembapan ini menggunakan algoritma Random Forest yang telah dilatih dengan data historis. Fitur input yang digunakan meliputi:
- Suhu rata-rata harian
- Curah hujan
- Lama penyinaran matahari
- Data historis 1 hari sebelumnya
- Rata-rata data 3 hari sebelumnya
- Informasi waktu (hari, bulan, tahun)
""")



# Add footer
st.markdown("-----------")
st.markdown("""
<div style="text-align: center">
    <p>Visualisasi dan Prediksi Data Cuaca Rangkaian Waktu</p>
    <p>© 2025 - Mohammad Iqbal Maulana</p>
</div>
""", unsafe_allow_html=True)