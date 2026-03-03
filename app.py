import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Streamlit Portfolio", layout="wide")


MODEL_PATH = "house_price_model.joblib"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


st.title("Streamlit Portfolio")

st.markdown("""
Selamat datang di portofolio saya.  
Aplikasi ini menampilkan profil singkat, proyek data science & machine learning yang pernah saya kerjakan,
serta demo implementasi prediksi berbasis model machine learning.
""")

st.divider()

st.header("👤 Tentang Saya")

st.markdown("""
**Nama:** Erwin M 
**Latar Belakang:** Technical Consultant

**Keahlian:**
- Python  
- Machine Learning  
- Computer Vision
""")

st.divider()

st.header("Proyek Saya")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("House Price Prediction")
    st.image("project1.png", caption="House Price Prediction", width=300)
    st.write("""
    Prediksi harga rumah menggunakan machine learning
    (Linear Regression, Random Forest, XGBoost).
    Dataset berasal dari kompetisi Kaggle (Ames Housing).
    """)
    st.button("Lihat Detail Proyek", key="proj1")

with col2:
    st.subheader("Chatbot RAG FC Barcelona & Indofood")
    st.image("project2.png", caption="Project 2", width=300)
    st.write("Chatbot RAG yang memuat laporan peforma FC Barcelona musim 2024/25 dan laporan tahunan Indofood pada tahun 2025 (Per September).")
    st.markdown(
        '<a href="https://simple-rag-hatfan.streamlit.app/" target="_blank">'
        '<button style="padding:8px 16px;">Lihat Detail Proyek</button>'
        '</a>',
        unsafe_allow_html=True
    )

with col3:
    st.subheader("Rendang GPT")
    st.image("project3.png", caption="Project 3", width=300)
    st.write("Chatbot pemberi resep rendang")
    st.markdown(
        '<a href="https://rendang-gpt2-hatfan.streamlit.app/" target="_blank">'
        '<button style="padding:8px 16px;">Lihat Detail Proyek</button>'
        '</a>',
        unsafe_allow_html=True
    )

st.divider()

st.header("Demo Prediksi Berbasis Model")

if model is not None:
    st.success("Model machine learning terintegrasi dan siap digunakan")
else:
    st.warning("Model belum terdeteksi. Pastikan file model tersedia.")

st.markdown("""
Upload file CSV untuk melakukan prediksi harga rumah.
File CSV **harus memiliki struktur kolom yang sama** seperti data training
(tanpa kolom `SalePrice`).
""")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("Preview Data Input")
    st.dataframe(df_input.head())

    if st.button("Jalankan Prediksi"):
        if model is None:
            st.error("Model belum tersedia. Tidak bisa menjalankan prediksi.")
        else:
            with st.spinner("Menjalankan prediksi..."):
                preds = np.expm1(model.predict(df_input))

            df_result = df_input.copy()
            df_result["PredictedSalePrice"] = preds.astype(int)

            st.session_state["prediction_result"] = df_result

            st.subheader("Hasil Prediksi")
            st.dataframe(df_result.head())

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil Prediksi",
                csv,
                "house_price_predictions.csv",
                "text/csv"
            )

st.divider()

st.header("Visualisasi Data & Model")

if "prediction_result" not in st.session_state:
    st.info("Silakan jalankan prediksi terlebih dahulu untuk melihat visualisasi.")
else:
    df_vis = st.session_state["prediction_result"]

    visual_option = st.selectbox(
        "Pilih visualisasi",
        [
            "Distribusi Harga Prediksi",
            "Korelasi Fitur Numerik vs Harga",
            "Ringkasan Prediksi"
        ]
    )

    if visual_option == "Distribusi Harga Prediksi":
        fig, ax = plt.subplots()
        sns.histplot(df_vis["PredictedSalePrice"], kde=True, ax=ax)
        ax.set_title("Distribusi Predicted Sale Price")
        st.pyplot(fig)

    elif visual_option == "Korelasi Fitur Numerik vs Harga":
        numeric_cols = df_vis.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 1:
            corr = df_vis[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Tidak cukup fitur numerik untuk korelasi.")

    elif visual_option == "Ringkasan Prediksi":
        st.metric("Jumlah Data", len(df_vis))
        st.metric(
            "Harga Minimum",
            f"{df_vis['PredictedSalePrice'].min():,.0f}"
        )
        st.metric(
            "Harga Rata-rata",
            f"{df_vis['PredictedSalePrice'].mean():,.0f}"
        )
        st.metric(
            "Harga Maksimum",
            f"{df_vis['PredictedSalePrice'].max():,.0f}"
        )

st.divider()

st.caption("© 2026 - Streamlit Portfolio")
