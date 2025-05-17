import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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



# Visualisasi Pola Cuaca dengan Tab
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


# Header dengan pemisah
st.markdown("---")
st.markdown(
    "<h2 style='text-align: center;'>🔢 Pembagian Dataset Training dan Testing 🔢</h2>",
    unsafe_allow_html=True
)

# Muat data dari file
X_train, X_test, y_train, y_test = joblib.load("./Dataset/split_data_joblib.pkl")

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



# Load model .joblib
model_rf = joblib.load("./Dataset/model_kelembapan_rf.joblib")

model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

# Tampilkan metrik evaluasi
st.markdown("---")
st.markdown("<h2 style='text-align: center;'> Model Random Forest</h2>", unsafe_allow_html=True)

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


# Perbandingan Nilai Aktual dan Prediksi dengan Plotly
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


y_pred_gb = joblib.load("./Dataset/model_kelembapan_gb.joblib")

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = np.mean(np.abs((y_test - y_pred_gb) / y_test)) * 100

st.markdown("---")
st.markdown("<h2 style='text-align: center;'> Model Gradient Boosting </h2>", unsafe_allow_html=True)

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


y_pred_xgb = joblib.load("./Dataset/model_kelembapan_xgb.joblib")
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

st.markdown("---")
st.markdown("<h2 style='text-align: center;'> Model XGBoost </h2>", unsafe_allow_html=True)

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

st.markdown("---")
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

with st.expander("📘 Penjelasan Metrik Evaluasi"):
    st.markdown("""
    - **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara prediksi dan nilai aktual.
    - **MSE (Mean Squared Error)**: Rata-rata dari kuadrat selisih antara prediksi dan nilai aktual.
    - **RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE, lebih sensitif terhadap outlier.
    - **R² (R-squared)**: Seberapa baik model menjelaskan variansi data (semakin mendekati 1, semakin baik).
    - **MAPE (Mean Absolute Percentage Error)**: Persentase rata-rata kesalahan prediksi terhadap nilai aktual.
    """)






# Add footer
st.markdown("-----------")
st.markdown("""
<div style="text-align: center">
    <p>Visualisasi dan Prediksi Data Cuaca Rangkaian Waktu</p>
    <p>© 2025 - Mohammad Iqbal Maulana</p>
</div>
""", unsafe_allow_html=True)