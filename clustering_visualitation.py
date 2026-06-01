import streamlit as st
import pandas as pd
import joblib

# =====================================
# LOAD DATA & MODEL
# =====================================

df = pd.read_csv("Data_Tingkat_Kemiskinan.csv")

model = joblib.load("clustering_kemiskinan.pkl")

# Jika saat training memakai scaler
# scaler = joblib.load("scaler.pkl")

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Prediksi Cluster Kemiskinan",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Prediksi Cluster Wilayah Kemiskinan")

st.markdown(
    """
    Pilih kabupaten/kota kemudian isi atau ubah nilai indikator
    sosial ekonomi untuk melihat hasil cluster wilayah.
    """
)

# =====================================
# DROPDOWN WILAYAH
# =====================================

daftar_wilayah = sorted(df["Kabupaten_Kota"].unique())

wilayah = st.selectbox(
    "Pilih Kabupaten / Kota",
    daftar_wilayah
)

# =====================================
# AMBIL DATA DEFAULT
# =====================================

data_default = df[
    df["Kabupaten_Kota"] == wilayah
].iloc[0]

# =====================================
# INPUT FITUR
# =====================================

st.subheader("Input Data Wilayah")

col1, col2 = st.columns(2)

with col1:

    rls = st.number_input(
        "Rata-rata Lama Sekolah",
        value=float(data_default["Rata-rata Lama Sekolah"])
    )

    ipg = st.number_input(
        "Indeks Pembangunan Gender",
        value=float(data_default["Indeks Pembangunan Gender"])
    )

    uhh = st.number_input(
        "Usia Harapan Hidup",
        value=float(data_default["Usia Harapan Hidup"])
    )

    pengeluaran = st.number_input(
        "Pengeluaran Per Kapita",
        value=float(data_default["PengeluaranPerKapita"])
    )

with col2:

    pdrb = st.number_input(
        "Produk Domestik Regional Bruto",
        value=float(data_default["Produk Domestik Regional Bruto"])
    )

    ikk = st.number_input(
        "Indeks Kemahalan Konstruksi",
        value=float(data_default["Indeks Kemahalan Konstruksi"])
    )

    rokok = st.number_input(
        "Pengeluaran Perkapita Rokok",
        value=float(data_default["PengeluaranPerkapita_Rokok"])
    )

# =====================================
# PREDIKSI
# =====================================

if st.button("🔍 Prediksi Cluster"):

    data_baru = pd.DataFrame(
        [[
            rls,
            ipg,
            uhh,
            pengeluaran,
            pdrb,
            ikk,
            rokok
        ]],
        columns=[
            "Rata-rata Lama Sekolah",
            "Indeks Pembangunan Gender",
            "Usia Harapan Hidup",
            "PengeluaranPerKapita",
            "Produk Domestik Regional Bruto",
            "Indeks Kemahalan Konstruksi",
            "PengeluaranPerkapita_Rokok"
        ]
    )

    # Jika training menggunakan scaler
    # data_baru = scaler.transform(data_baru)

    cluster = model.predict(data_baru)

    hasil = int(cluster[0])

    st.success(
        f"Hasil Prediksi: Cluster {hasil}"
    )

    # =====================================
    # INTERPRETASI CLUSTER
    # =====================================

    if hasil == 0:
        st.info(
            "Cluster 0"
        )

    elif hasil == 1:
        st.info(
            "Cluster 1"
        )

    elif hasil == 2:
        st.info(
            "Cluster 2"
        )

    st.markdown("---")

    st.subheader("Ringkasan")

    st.write(f"**Wilayah :** {wilayah}")
    st.write(f"**Cluster :** {hasil}")