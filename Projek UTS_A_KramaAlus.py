import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# FUNCTION PREPROCESSING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()

# ==========================================
# STREAMLIT UPLOAD
# ==========================================
st.title('Prediksi Klasifikasi Teks')

# Upload dataset
train_file = st.file_uploader("Upload Train Dataset", type=["csv"])
valid_file = st.file_uploader("Upload Valid Dataset", type=["csv"])
test_file = st.file_uploader("Upload Test Dataset", type=["csv"])

if train_file is not None and valid_file is not None and test_file is not None:
    # Load the datasets
    train = pd.read_csv(train_file)
    valid = pd.read_csv(valid_file)
    test = pd.read_csv(test_file)
    
    # Gabungkan train dan valid
    data = pd.concat([train, valid], axis=0).reset_index(drop=True)

    st.write("Data Train + Validasi:", data.shape)
    st.write("Data Test:", test.shape)

    # ==========================================
    # EDA (Visualisasi Data)
    # ==========================================
    data['text_length'] = data['sentence'].apply(lambda x: len(str(x)))
    st.subheader('Distribusi Panjang Teks')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(data['text_length'], ax=ax1)
    st.pyplot(fig1)

    data['word_count'] = data['sentence'].apply(lambda x: len(str(x).split()))
    st.subheader('Distribusi Jumlah Kata')
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data['word_count'], ax=ax2)
    st.pyplot(fig2)

    label_counts = data['fuel'].value_counts()
    st.subheader('Jumlah Teks per Label')
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # ==========================================
    # PREPROCESSING DATA
    # ==========================================
    data['sentence'] = data['sentence'].apply(clean_text)
    test['sentence'] = test['sentence'].apply(clean_text)

    # Encode label
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['sentence'].apply(lambda x: x.split(',')))

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['sentence'])
    X_test_final = vectorizer.transform(test['sentence'])

    # Latih Model
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X, y)

    model_nb = MultinomialNB()
    model_nb.fit(X, y)

    # ==========================================
    # FINAL TESTING (DATA TEST)
    # ==========================================
    y_test_pred_lr = model_lr.predict(X_test_final)
    y_test_pred_nb = model_nb.predict(X_test_final)

    hasil_test_lr = pd.DataFrame({
        'teks': test['sentence'],
        'prediksi_label_lr': y_test_pred_lr
    })

    hasil_test_nb = pd.DataFrame({
        'teks': test['sentence'],
        'prediksi_label_nb': y_test_pred_nb
    })

    st.subheader("Prediksi dengan Logistic Regression")
    st.write(hasil_test_lr.head())

    st.subheader("Prediksi dengan Naive Bayes")
    st.write(hasil_test_nb.head())

    # ==========================================
    # PREDIKSI TEKS BARU (OPSIONAL)
    # ==========================================
    def prediksi_teks_baru(teks_input):
        teks_bersih = clean_text(teks_input)  # Bersihkan teks
        teks_vec = vectorizer.transform([teks_bersih])  # Vektorisasi teks baru
        pred = model_lr.predict(teks_vec)  # Prediksi label
        return pred  # Langsung kembalikan prediksi tanpa inverse_transform

    teks_baru = st.text_input("Masukkan teks untuk prediksi:")
    if teks_baru:
        hasil_prediksi = prediksi_teks_baru(teks_baru)
        st.write("Prediksi untuk teks baru:", hasil_prediksi)

else:
    st.write("Silakan upload file dataset terlebih dahulu.")
