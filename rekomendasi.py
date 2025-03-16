import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# Header
st.markdown("<h1 style='text-align: center;'>🌱 Dashboard Rekomendasi Tanaman 🔍</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>
Selamat datang di Dashboard Rekomendasi Tanaman. Dashboard ini dirancang untuk memberikan rekomendasi tanaman yang paling sesuai dengan kondisi lingkungan Anda. Dengan menganalisis berbagai parameter seperti suhu, kelembapan, curah hujan, dan pH tanah, sistem kami menggunakan teknologi machine learning dan algoritma prediksi untuk menentukan tanaman yang optimal.
<br><br>
Setiap kondisi lingkungan memiliki karakteristik unik yang mempengaruhi pertumbuhan tanaman. Oleh karena itu, rekomendasi yang kami berikan didasarkan pada analisis mendalam dan data akurat, sehingga Anda dapat mengambil keputusan yang tepat dalam memilih tanaman untuk mencapai hasil pertanian yang maksimal.
<br><br>
Silakan masukkan parameter lingkungan Anda dan dapatkan rekomendasi tanaman yang sesuai, dan biarkan sistem kami membantu Anda menemukan solusi terbaik untuk kebutuhan pertanian Anda. Selamat mencoba dan semoga sukses!
<br><br><br><br>
</p>
""", unsafe_allow_html=True)

# Ui Dataset
def load_data():
    df = pd.read_csv("./Dataset/Crop_recommendation_ID.csv")
    return df
df = load_data()

# Buat dua kolom utama
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3 style='text-align: center;'>🔍 Informasi Dataset</h3>", unsafe_allow_html=True)
    selected_columns = st.multiselect("Pilih Kolom yang Ditampilkan", df.columns, default=df.columns)
    if selected_columns:
        search_col = st.selectbox("Pilih Kolom untuk Pencarian", selected_columns)
    else:
        search_col = None
    search_value = st.text_input("Masukkan Nilai yang Dicari")

    # Filter dataset berdasarkan kolom yang dipilih
    filtered_df = df[selected_columns] if selected_columns else df

    # Jika ada pencarian, filter data berdasarkan nilai yang dicari
    if search_value and search_col:
        filtered_df = filtered_df[filtered_df[search_col].astype(str).str.contains(search_value, case=False, na=False)]

    # Membulatkan angka ke atas
    filtered_df = filtered_df.applymap(lambda x: np.ceil(x) if isinstance(x, (int, float)) else x)
    st.write(f"**Jumlah Baris :** {filtered_df.shape[0]}")
    st.write(f"**Jumlah Kolom :** {filtered_df.shape[1]}")

with col2:
    st.markdown("<h3 style='text-align: center;'>📋 Tampilan Dataset</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df, height=400, width=1500)

# Visualisasi
st.markdown(
    """
    <h2 style='text-align: center;'>📈 Analisis Sebaran Variabel</h2>
    <p style='text-align: justify; text-indent: 50px;'>Visualisasi sebaran variabel digunakan untuk memahami bagaimana distribusi data dari setiap fitur dalam dataset. Dengan melihat distribusi nilai, kita dapat mengidentifikasi pola, tren, serta kemungkinan adanya outlier atau anomali dalam data. Melalui histogram yang ditampilkan, kita dapat mengetahui apakah suatu variabel memiliki distribusi normal, skewed (miring), atau terdapat nilai ekstrem yang dapat mempengaruhi analisis lebih lanjut. Selain itu, dengan menerapkan filter pada dataset, kita dapat membandingkan distribusi data sebelum dan sesudah penyaringan untuk memahami bagaimana data berubah berdasarkan kriteria yang dipilih. Analisis ini sangat penting dalam eksplorasi data awal sebelum dilakukan pemodelan atau pengambilan keputusan, terutama dalam bidang pertanian berbasis data, di mana pemahaman terhadap karakteristik tanah, curah hujan, dan faktor lingkungan lainnya sangat berpengaruh terhadap rekomendasi tanaman yang optimal. 🚀🌱</p>
    """,
    unsafe_allow_html=True
)

selected_vars = st.multiselect(
    "Pilih Variabel untuk Distribusi",
    ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    default=['N', 'P', 'K']
)

if selected_vars and not filtered_df.empty:
    n_vars = len(selected_vars)
    cols = 3
    rows = (n_vars // cols) + (n_vars % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i, var in enumerate(selected_vars):
        sns.histplot(filtered_df[var], bins=20, kde=True, ax=axes[i], color=np.random.rand(3,))
        axes[i].set_title(f"Sebaran {var} ", fontsize=12)
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Frekuensi")

    # Hapus subplot kosong jika ada
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Tampilkan plot di Streamlit
    st.pyplot(fig)
else:
    st.warning("Pilih minimal satu variabel dan pastikan dataset tidak kosong untuk menampilkan sebaran data.")


# 🟢 **PAIRPLOT (Visualisasi Hubungan Antar Variabel)**
st.markdown(
    """
    <h2 style='text-align: center;'>🌿 Visualisasi Hubungan Antar Variabel</h2>
    <p style='text-align: justify; text-indent: 50px;'>
    Dalam analisis data pertanian, memahami hubungan antara variabel lingkungan sangat penting untuk menentukan faktor-faktor yang paling berpengaruh terhadap pertumbuhan tanaman. Oleh karena itu, visualisasi menggunakan <b>Pairplot</b> memungkinkan kita untuk melihat korelasi antar variabel serta bagaimana setiap parameter lingkungan berinteraksi satu sama lain dalam menentukan jenis tanaman yang optimal. Dengan menyusun scatter plot antara dua variabel dan histogram distribusinya, kita dapat mengidentifikasi pola hubungan seperti korelasi positif atau negatif yang dapat membantu dalam pengambilan keputusan berbasis data.
    </p>
    
    <p style='text-align: justify; text-indent: 50px;'>
    Misalnya, kadar nitrogen, fosfor, dan kalium dalam tanah mungkin menunjukkan korelasi tertentu dengan tingkat pH atau curah hujan. Jika tanaman tertentu cenderung tumbuh lebih baik pada kombinasi kadar unsur hara dan tingkat kelembapan tertentu, maka pola ini dapat dikenali melalui visualisasi Pairplot. Selain itu, dengan menggunakan warna berbeda berdasarkan label tanaman, kita dapat melihat bagaimana setiap jenis tanaman memiliki distribusi unik dalam ruang fitur tersebut. 
    </p>

    <p style='text-align: justify; text-indent: 50px;'>
    Dalam dashboard ini, pengguna dapat memilih variabel lingkungan yang ingin dianalisis serta menyaring data berdasarkan jenis tanaman tertentu. Untuk meningkatkan efisiensi, jumlah sampel yang ditampilkan dibatasi agar analisis tetap berjalan dengan lancar tanpa mengorbankan representasi data. Dengan pendekatan ini, kita dapat lebih mudah menemukan hubungan signifikan yang dapat membantu petani atau peneliti dalam menentukan jenis tanaman yang paling sesuai dengan kondisi lingkungan tertentu. 🌱📊
    </p>
    """,
    unsafe_allow_html=True
)

# Buat container untuk filter
with st.expander("🔍 Filter dan Pengaturan Visualisasi", expanded=True):
    filter_col1, filter_col2 = st.columns([1, 1])
    
    with filter_col1:
        st.subheader("Pemilihan Variabel")
        selected_pairplot = st.multiselect(
            "Pilih Variabel untuk Pairplot",
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
            default=['N', 'P', 'K'],
            help="Pilih variabel yang ingin Anda bandingkan dalam visualisasi"
        )
        # Pengaturan ukuran plot
        plot_height = st.number_input(
            "Ukuran Plot",
            min_value=1,
            max_value=5,
            value=3,
            help="Atur ukuran setiap subplot dalam pairplot"
        )
    
    with filter_col2:
        st.subheader("Filter Data")
        # Pastikan kolom label ada
        if "label" not in df.columns:
            st.error("Kolom 'label' tidak ditemukan dalam dataset! Periksa nama kolom.")
            st.stop()
        
        # Pilihan label tanaman dengan fitur pencarian
        all_labels = df["label"].unique().tolist()
        search_label = st.text_input(
            "Cari Label Tanaman",
            help="Ketik untuk mencari label tanaman tertentu"
        )
        
        # Filter label berdasarkan pencarian
        if search_label:
            filtered_labels = [label for label in all_labels if search_label.lower() in label.lower()]
        else:
            filtered_labels = all_labels
            
        selected_labels = st.multiselect(
            "Pilih Label Tanaman untuk Ditampilkan",
            options=filtered_labels,
            default=filtered_labels[:3],
            help="Pilih tanaman yang ingin Anda bandingkan"
        )

if selected_labels:
    with st.expander("⚙️ Pengaturan Sampel per Label", expanded=True):
        st.subheader("Pengaturan Sampel per Label")
        
        # Dictionary untuk menyimpan jumlah sampel per label
        samples_per_label = {}
        
        # Buat kolom untuk pengaturan sampel
        col_count = 4 
        cols = st.columns(col_count)
        
        for idx, label in enumerate(selected_labels):
            col_idx = idx % col_count
            with cols[col_idx]:
                available_samples = len(df[df['label'] == label])
                st.write(f"**{label}**")
                
                # Input untuk mengatur jumlah sampel
                n_samples = st.number_input(
                    f"Jumlah sampel untuk {label}",
                    min_value=1,
                    max_value=int(available_samples),
                    value=min(100, int(available_samples)),
                    help=f"Masukkan jumlah sampel untuk {label}"
                )
                samples_per_label[label] = n_samples
                
                # Tampilkan jumlah data yang diambil
                st.write(f"Data diambil: {n_samples} dari {available_samples}")
        
        # Proses sampling data
        sampled_dfs = []
        for label, n_samples in samples_per_label.items():
            label_data = df[df['label'] == label]
            if len(label_data) > n_samples:
                sampled_data = label_data.sample(n_samples, random_state=42)
            else:
                sampled_data = label_data
            sampled_dfs.append(sampled_data)
        
        # Gabungkan semua data yang telah disampling
        filtered_df = pd.concat(sampled_dfs)
        
        # Informasi hasil sampling
        st.write("---")
        total_samples = sum(samples_per_label.values())
        st.metric(
            "Total Data Setelah Sampling",
            f"{len(filtered_df):,} baris"
        )

# Buat visualisasi jika data tersedia
total_samples = sum(samples_per_label.values())
if selected_pairplot and not filtered_df.empty:
    with st.spinner("Membuat visualisasi..."):
        sns.set_theme(style="whitegrid")
        
        # Buat Pairplot dengan pengaturan yang ditingkatkan
        pairplot_fig = sns.pairplot(
            filtered_df,
            vars=selected_pairplot,
            hue="label",
            # diag_kind='hist',
            corner=True,
            height=plot_height,
            plot_kws={'alpha': 0.6},
            diag_kws={'alpha': 0.6}
        )
        
        # Atur judul
        pairplot_fig.fig.suptitle(
            "Visualisasi Hubungan Antar Variabel",
            y=1.02,
            fontsize=16
        )
        
        # Tambahkan label pada setiap subplot
        for i, var1 in enumerate(selected_pairplot):
            for j, var2 in enumerate(selected_pairplot):
                if j < i:
                    ax = pairplot_fig.axes[i][j]
                    ax.set_xlabel(f"{var2}")
                    ax.set_ylabel(f"{var1}")
                    if i == len(selected_pairplot)-1:
                        ax.set_xlabel(f"{var2}")
        
        # Tampilkan plot
        st.pyplot(pairplot_fig)
else:
    st.warning("Pilih minimal satu label tanaman untuk menampilkan pengaturan sampel dan visualisasi.")
# 🔹 **Memuat Model yang Sudah Disimpan**
@st.cache_resource
def load_model():
    with open("./model_RandomForest copy.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model_randomForest = load_model()

# 🔹 **Memuat Data Uji dari File CSV**
X_test = pd.read_csv("./Dataset/X_test.csv")
y_test = pd.read_csv("./Dataset/y_test.csv").values.ravel()

# 🔹 **Melakukan Prediksi dengan Model**
y_pred = model_randomForest.predict(X_test)

# 🔹 **Evaluasi Model**
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# 🔹 **Mapping Label Encoding ke Nama Tanaman**
label_mapping = {
    'padi': 0,
    'jagung': 1,
    'buncis': 2,
    'kacang merah': 3,
    'kacang polong': 4,
    'kacang panjang': 5,
    'kacang hijau': 6,
    'kacang hitam': 7,
    'lentil': 8,
    'delima': 9,
    'pisang': 10,
    'mangga': 11,
    'anggur': 12,
    'semangka': 13,
    'melon': 14,
    'apel': 15,
    'jeruk': 16,
    'pepaya': 17,
    'kelapa': 18,
    'kapas': 19,
    'goni': 20,
    'kopi': 21
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# 🔹 **Konversi y_test ke nama label**
y_test_labels = [reverse_label_mapping[label] for label in y_test]
y_pred_labels = [reverse_label_mapping[label] for label in y_pred]

# 🔹 **Mengonversi Classification Report ke DataFrame**
def classification_report_to_df(y_test_labels, y_pred_labels):
    report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    df = pd.DataFrame(report_dict).T
    df = df.drop(index=["accuracy"], errors="ignore")
    df[["precision", "recall", "f1-score"]] = (df[["precision", "recall", "f1-score"]] * 100).round(0).astype(int)
    df["support"] = df["support"].astype(int)
    df = df.reset_index().rename(columns={"index": "Label"})
    return df


# 🔹 **Menampilkan Accuracy Score dalam Persen**
st.markdown("<h1 style='text-align: center;'> <br><br>🌱 Model Evaluasi Kinerja</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: green;'>✅ Accuracy Score: {accuracy * 100:.2f}%</h3>", unsafe_allow_html=True)

# Konversi classification report ke DataFrame
report_df = classification_report_to_df(y_test_labels, y_pred_labels)

# Atur ukuran kolom agar lebih seimbang
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("<h2 style='text-align: center;'>📊 Classification Report</h2>", unsafe_allow_html=True)
    # Menampilkan dataframe dengan label tanaman dan nilai bulat
    st.dataframe(report_df, height=600, use_container_width=True)

with col2:
    st.markdown("<h2 style='text-align: center;'>🔎 Confusion Matrix</h2>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    ax.set_title("Confusion Matrix - Random Forest", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)


# 🔹 **Memuat Model dari File .pkl**
@st.cache_resource
def load_model():
    model_path = "./model_RandomForest copy.pkl" 
    
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        raise e

# 🔹 **Panggil model dari file**
model_randomForest = load_model()

# 🔹 **Tampilan Judul**
st.markdown("<h1 style='text-align: center;'><br><br>🌱 Prediksi Rekomendasi Tanaman 🔍</h1>", unsafe_allow_html=True)

# 🔹 **Form Input Data**
st.markdown("<h2 style='text-align: center;'>📝 Masukkan Data Lingkungan</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Kandungan Nitrogen (N)", min_value=0, value=50)
    P = st.number_input("Kandungan Fosfor (P)", min_value=0, value=20)
    K = st.number_input("Kandungan Kalium (K)", min_value=0, value=30)

with col2:
    temperature = st.number_input("Suhu (°C)", min_value=0.0, value=25.0)
    humidity = st.number_input("Kelembapan (%)", min_value=0, value=70)

with col3:
    ph = st.number_input("pH Tanah", min_value=0.0, value=6.5)
    rainfall = st.number_input("Curah Hujan (mm)", min_value=0, value=200)

# 🔹 **Dictionary untuk label tanaman**
label_predict = {
    'padi': 0,
    'jagung': 1,
    'buncis': 2,
    'kacang merah': 3,
    'kacang polong': 4,
    'kacang panjang': 5,
    'kacang hijau': 6,
    'kacang hitam': 7,
    'lentil': 8,
    'delima': 9,
    'pisang': 10,
    'mangga': 11,
    'anggur': 12,
    'semangka': 13,
    'melon': 14,
    'apel': 15,
    'jeruk': 16,
    'pepaya': 17,
    'kelapa': 18,
    'kapas': 19,
    'goni': 20,
    'kopi': 21
}

# 🔹 **Membalik dictionary untuk mempermudah pencarian nama berdasarkan angka**
label_reverse = {v: k for k, v in label_predict.items()}

# 🔹 **Prediksi dari Model yang Sudah Ada**
if st.button("🔎 Prediksi Tanaman"):
    # Input data yang diberikan
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Mengecek apakah input data kosong atau tidak
    if any(val is None or val == 0 for val in input_data[0]):
        st.error("Data yang dimasukkan tidak lengkap atau invalid.")
    else:
        # Prediksi menggunakan model Random Forest
        prediksi = model_randomForest.predict(input_data)[0]

        # Mendapatkan nama tanaman berdasarkan prediksi
        predicted_label = label_reverse.get(prediksi, "Tanaman Tidak Dikenal")

        # 🔹 **Menampilkan Hasil Prediksi**
        st.markdown("<h2 style='text-align: center;'>🌾 Hasil Prediksi Tanaman 🌾</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: green;'>🌿 {predicted_label.upper()} 🌿</h3>", unsafe_allow_html=True)