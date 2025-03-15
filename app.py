import streamlit as st

rekomendasi = st.Page("rekomendasi.py", title="Rekomendasi Tanaman", icon="🌱")
cuaca = st.Page("cuaca.py", title="cuaca", icon="🔍")
pg= st.navigation([rekomendasi, cuaca])
pg.run()
