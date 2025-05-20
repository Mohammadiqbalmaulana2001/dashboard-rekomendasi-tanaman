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

st.markdown("<h1 style='text-align: center;'>🌧️Peramalan Curah Hujan🌧️</h1>", unsafe_allow_html=True)
# Deskripsi
st.markdown("""
<div style='text-align: justify; font-size: 16px; text-indent: 40px; padding-bottom: 50px;'>
    Pada halaman ini, kita akan melakukan analisis data deret waktu (time series) cuaca untuk memprediksi curah hujan (<code>RR</code>) berdasarkan sejumlah parameter meteorologi. 
    Tujuan utama dari analisis ini adalah membangun model machine learning, seperti <b>Random Forest</b>, yang dapat memperkirakan besaran curah hujan berdasarkan variabel-variabel seperti suhu, kelembapan, dan intensitas sinar matahari.
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