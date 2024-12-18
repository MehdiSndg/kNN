import streamlit as st
import numpy as np
import pandas as pd

st.title("k-Nearest Neighbors Web Arayüzü (Streamlit)")

# Veri Kaynağı Seçimi
data_source = st.radio("Veri Giriş Yöntemi Seçin:", ["Manuel Veri Girişi", "CSV Yükle", "XLSX Yükle"])

if "rows" not in st.session_state:
    st.session_state.rows = 5
if "cols" not in st.session_state:
    st.session_state.cols = 4

uploaded_data = None
data_entries = None

if data_source == "Manuel Veri Girişi":
    st.write("Aşağıdaki tabloya verilerinizi girin. Son sütunun sınıf etiketi olduğunu varsayıyoruz.")

    # Butonlar: Satır/Sütun Ekle - Azalt
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("+ Satır Ekle"):
            st.session_state.rows += 1
    with col2:
        if st.button("- Satır Sil"):
            if st.session_state.rows > 1:
                st.session_state.rows -= 1
            else:
                st.warning("En az bir satır olmalıdır.")
    with col3:
        if st.button("+ Sütun Ekle"):
            st.session_state.cols += 1
    with col4:
        if st.button("- Sütun Sil"):
            if st.session_state.cols > 1:
                st.session_state.cols -= 1
            else:
                st.warning("En az bir sütun olmalıdır.")

    # Tabloyu oluştur
    data_entries = []
    for r in range(st.session_state.rows):
        row_cols = st.columns(st.session_state.cols)
        row_inputs = []
        for c in range(st.session_state.cols):
            cell_key = f"cell_{r}_{c}"
            val = row_cols[c].text_input(
                label="",
                key=cell_key,
                placeholder=f"({r},{c})"
            )
            row_inputs.append(val)
        data_entries.append(row_inputs)

elif data_source == "CSV Yükle":
    st.write("CSV formatında bir dosya yükleyin. Son sütunun sınıf etiketi olduğunu varsayıyoruz.")
    uploaded_file = st.file_uploader("CSV Dosyası Seçin", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Yüklenen Veri:")
            st.dataframe(df.head())
            uploaded_data = df.values
        except Exception as e:
            st.error(f"Dosya okunurken hata oluştu: {e}")

elif data_source == "XLSX Yükle":
    st.write("XLSX formatında bir dosya yükleyin. Son sütunun sınıf etiketi olduğunu varsayıyoruz.")
    uploaded_file = st.file_uploader("XLSX Dosyası Seçin", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)  # openpyxl kurulu olmalı.
            st.write("Yüklenen Veri:")
            st.dataframe(df.head(10))
            uploaded_data = df.values
        except Exception as e:
            st.error(f"Dosya okunurken hata oluştu: {e}")

st.write("### k değeri ve Sorgu Noktası")
k_value = st.text_input("k değeri", value="3")
query_str = st.text_input("Sorgu Noktası (virgülle ayrılacak)", value="")

def knn_predict(X, y, query, k=3):
    distances = np.sqrt(np.sum((X - query)**2, axis=1))
    idx = np.argsort(distances)[:k]
    classes, counts = np.unique(y[idx], return_counts=True)
    return classes[np.argmax(counts)]

if st.button("Tahmin Et"):
    # Veri Hazırlama
    if data_source == "Manuel Veri Girişi":
        for r in range(st.session_state.rows):
            for c in range(st.session_state.cols):
                if data_entries[r][c].strip() == "":
                    st.error("Tabloda boş hücre var, lütfen tüm hücreleri doldurun.")
                    st.stop()
        data = np.array(data_entries)
    else:
        # CSV veya XLSX
        if uploaded_data is None:
            st.error("Lütfen geçerli bir dosya yükleyin.")
            st.stop()
        data = uploaded_data

    try:
        k = int(k_value)
    except:
        st.error("k değeri geçerli bir tamsayı olmalıdır.")
        st.stop()

    if query_str.strip() == "":
        st.error("Sorgu noktası boş!")
        st.stop()

    try:
        query = np.array([float(x) for x in query_str.split(",")])
    except:
        st.error("Sorgu noktası formatı hatalı. Virgülle ayrılmış sayılar girin.")
        st.stop()

    if data.shape[1] < 2:
        st.error("Veri en az 1 özellik ve 1 sınıf sütunu içermelidir.")
        st.stop()

    X = data[:, :-1].astype(float)
    y = data[:, -1]

    if query.shape[0] != X.shape[1]:
        st.error("Sorgu noktası boyutu veri boyutuyla eşleşmiyor!")
        st.stop()

    pred = knn_predict(X, y, query, k)
    st.success(f"Tahmin Edilen Sınıf: {pred}")
