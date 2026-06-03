import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Cluster Kemiskinan",
    page_icon="📊",
    layout="wide"
)

df = pd.read_csv("Data_Tingkat_Kemiskinan.csv")
model = joblib.load("clustering_kemiskinan (5).pkl")

st.title("Prediksi Cluster Wilayah Kemiskinan")

st.markdown("""
Aplikasi ini digunakan untuk mengelompokkan kabupaten/kota berdasarkan karakteristik sosial ekonomi di Indonesia tahun 2024 yang bersumber dari Badan Pusat Statistik (BPS) menggunakan algoritma **K-Means Clustering**.

### Interpretasi Cluster
🟢 **Cluster 0** → Wilayah Relatif Sejahtera 

🔴 **Cluster 1** → Wilayah Rentan Kemiskinan
""")

st.sidebar.header("Input Data Wilayah")

daftar_wilayah = sorted(df["Kabupaten_Kota"].unique())

wilayah = st.sidebar.selectbox(
    "Pilih Kabupaten / Kota",
    daftar_wilayah
)

data_default = df[
    df["Kabupaten_Kota"] == wilayah
].iloc[0]

rls = st.sidebar.number_input(
    "Rata-rata Lama Sekolah",
    value=float(data_default["Rata-rata Lama Sekolah"])
)

ipg = st.sidebar.number_input(
    "Indeks Pembangunan Gender",
    value=float(data_default["Indeks Pembangunan Gender"])
)

uhh = st.sidebar.number_input(
    "Usia Harapan Hidup",
    value=float(data_default["Usia Harapan Hidup"])
)

pengeluaran = st.sidebar.number_input(
    "Pengeluaran Per Kapita",
    value=float(data_default["PengeluaranPerKapita"])
)

pdrb = st.sidebar.number_input(
    "Produk Domestik Regional Bruto",
    value=float(data_default["Produk Domestik Regional Bruto"])
)

ikk = st.sidebar.number_input(
    "Indeks Kemahalan Konstruksi",
    value=float(data_default["Indeks Kemahalan Konstruksi"])
)

rokok = st.sidebar.number_input(
    "Pengeluaran Perkapita Rokok",
    value=float(data_default["PengeluaranPerkapita_Rokok"])
)

prediksi = st.sidebar.button("🔍 Prediksi Cluster")

if prediksi:

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

    cluster = model.predict(data_baru)
    hasil = int(cluster[0])

    st.markdown("---")

    st.subheader("📍 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Kabupaten / Kota",
            value=wilayah
        )

    with col2:
        st.metric(
            label="Cluster",
            value=f"Cluster {hasil}"
        )

    st.markdown("---")

    if hasil == 0:

        st.error("🔴 Cluster 0 - Wilayah Rentan Kemiskinan")

        st.markdown("""
        ### Interpretasi

        Wilayah ini memiliki karakteristik yang lebih rentan terhadap kemiskinan.""")

    elif hasil == 1:

        st.success("🟢 Cluster 1 - Wilayah Relatif Sejahtera")

        st.markdown("""
        ### Interpretasi

        Wilayah ini memiliki kondisi sosial ekonomi yang relatif lebih baik dibandingkan kelompok lainnya.

        Indikator pendidikan, kesehatan, dan kesejahteraan masyarakat cenderung menunjukkan kondisi yang lebih baik.

        Hasil ini menunjukkan bahwa wilayah tersebut memiliki karakteristik yang mirip dengan kelompok wilayah yang relatif lebih sejahtera.
        """)

    st.markdown("---")

    st.subheader("📋 Ringkasan Data Input")

    ringkasan = pd.DataFrame({
        "Variabel": [
            "Rata-rata Lama Sekolah",
            "Indeks Pembangunan Gender",
            "Usia Harapan Hidup",
            "Pengeluaran Per Kapita",
            "Produk Domestik Regional Bruto",
            "Indeks Kemahalan Konstruksi",
            "Pengeluaran Perkapita Rokok"
        ],
        "Nilai": [
            rls,
            ipg,
            uhh,
            pengeluaran,
            pdrb,
            ikk,
            rokok
        ]
    })

    st.dataframe(
        ringkasan,
        use_container_width=True
    )
