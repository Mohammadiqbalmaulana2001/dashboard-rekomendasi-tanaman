import streamlit as st

st.set_page_config(page_title="Aplikasi Pertanian", layout="wide", page_icon="🌿")

css = """
<style>
    section[data-testid="stSidebar"] {
        position: relative;
    }
    section[data-testid="stSidebar"]:before {
        content: "APLIKASI PERTANIAN";
        display: block;
        padding: 1.2rem 1rem;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.8rem;
    }
    /* Tambahkan juga subtitle jika diperlukan */
    section[data-testid="stSidebar"]:after {
        content: "Navigation Tanaman dan Cuaca";
        display: block;
        position: absolute;
        top: 3.5rem;
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
cuaca = st.Page("cuaca.py", title="Cuaca", icon="🔍")

# Buat navigasi
pg = st.navigation([cuaca,rekomendasi])
pg.run()