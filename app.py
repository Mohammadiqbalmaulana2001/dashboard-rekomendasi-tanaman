import streamlit as st

st.set_page_config(page_title="Aplikasi Pertanian", layout="wide", page_icon="🌿")

css = """
<style>
    section[data-testid="stSidebar"] {
        position: relative;
    }

    section[data-testid="stSidebar"]:before {
        content: "DASHBOARD PERTANIAN";
        display: block;
        padding: 1.2rem 1rem;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem; /* Tambahkan jarak bawah */
    }

    section[data-testid="stSidebar"]:after {
        content: "Navigasi Tanaman dan Cuaca";
        display: block;
        position: absolute;
        top: 5.3rem; /* Atur ulang posisi top agar tidak tabrakan */
        left: 0;
        right: 0;
        padding: 0 1rem 0.8rem;
        font-size: 0.9rem;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)
st.balloons()
# Definisikan halaman
rekomendasi = st.Page("rekomendasi.py", title="Rekomendasi Tanaman", icon="🌱")
kelembapan = st.Page("kelembapan.py", title="Kelembapan", icon="💧")
suhu = st.Page("suhu.py", title="Suhu", icon="🌡️")
curah_hujan = st.Page("curah_hujan.py", title="Curah Hujan", icon="🌧️")
# Buat navigasi
pg = st.navigation([suhu,kelembapan,curah_hujan,rekomendasi])
pg.run()