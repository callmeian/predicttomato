import streamlit as st
import pandas as pd
import pickle

# =========================
# Fungsi Load Model & Data
# =========================

@st.cache_resource
def load_model(path_model: str):
    """Memuat model terbaik yang sudah disimpan (pickle)."""
    with open(path_model, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    """Memuat dan menggabungkan 3 dataset tomat untuk kebutuhan informasi di aplikasi."""
    df_panen = pd.read_csv("luas_panen_tomat.csv")
    df_areal = pd.read_csv("luas_areal_tanam_tomat.csv")
    df_produksi = pd.read_csv("produksi_tomat.csv")

    df = df_areal.merge(
        df_panen[["kode_provinsi", "kode_kabupaten_kota", "tahun", "luas_panen"]],
        on=["kode_provinsi", "kode_kabupaten_kota", "tahun"],
        how="inner"
    )

    df = df.merge(
        df_produksi[["kode_provinsi", "kode_kabupaten_kota", "tahun", "produksi_tomat"]],
        on=["kode_provinsi", "kode_kabupaten_kota", "tahun"],
        how="inner"
    )
    return df

# =========================
# Main App
# =========================

def main():
    st.set_page_config(
        page_title="Prediksi Produksi Tomat Jawa Barat",
        layout="centered"
    )

    st.title("📈 Prediksi Produksi Tomat di Jawa Barat")
    st.write(
        "Aplikasi ini menggunakan model regresi terbaik (Linear Regression / SVR / Random Forest) "
        "yang telah dilatih untuk memprediksi **produksi tomat (ton)** berdasarkan luas areal tanam, "
        "luas panen, dan tahun di setiap kabupaten/kota Jawa Barat."
    )

    # Load data dan model
    df = load_data()
    model = load_model("model_regresi_produksi_tomat_terbaik.pkl")

    # =========================
    # Sidebar - Input Pengguna
    # =========================
    st.sidebar.header("Input Fitur Prediksi")


    kabupaten_list = sorted(df["nama_kabupaten_kota"].dropna().unique().tolist())
    default_kab = kabupaten_list[0] if kabupaten_list else ""

    nama_kabupaten = st.sidebar.selectbox(
        "Pilih Kabupaten/Kota",
        options=kabupaten_list,
        index=0 if default_kab in kabupaten_list else 0
    )

    # Ambil nilai default rentang dari dataset
    luas_areal_min = float(df["luas_areal_tanam"].min())
    luas_areal_max = float(df["luas_areal_tanam"].max())
    luas_panen_min = float(df["luas_panen"].min())
    luas_panen_max = float(df["luas_panen"].max())
    tahun_min = int(df["tahun"].min())
    tahun_max = int(df["tahun"].max())

    luas_areal_tanam = st.sidebar.number_input(
        "Luas Areal Tanam (ha)",
        min_value=0.0,
        max_value=luas_areal_max * 1.5,
        value=float(df["luas_areal_tanam"].median())
    )

    luas_panen = st.sidebar.number_input(
        "Luas Panen (ha)",
        min_value=0.0,
        max_value=luas_panen_max * 1.5,
        value=float(df["luas_panen"].median())
    )

    tahun = st.sidebar.number_input(
        "Tahun",
        min_value=tahun_min,
        max_value=tahun_max + 5,
        value=tahun_max
    )

    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Prediksi Produksi Tomat")

    # =========================
    # Tampilan Data (opsional)
    # =========================
    with st.expander("Lihat Data Ringkas"):
        st.write("Contoh 10 baris data gabungan:")
        st.dataframe(df.head(10))
        st.write("Statistik deskriptif fitur numerik:")
        st.dataframe(df[["luas_areal_tanam", "luas_panen", "produksi_tomat"]].describe())

    # =========================
    # Prediksi
    # =========================
    if predict_button:
        # Susun input ke dalam DataFrame (harus sama dengan saat training)
        input_data = pd.DataFrame([{
            "nama_kabupaten_kota": nama_kabupaten,
            "luas_areal_tanam": luas_areal_tanam,
            "luas_panen": luas_panen,
            "tahun": tahun
        }])

        try:
            pred = model.predict(input_data)
            pred_value = float(pred[0])

            st.subheader("Hasil Prediksi")
            st.success(f"Perkiraan produksi tomat: **{pred_value:,.2f} ton**")  # format ribuan & 2 desimal

            st.caption(
                "Catatan: Hasil prediksi bergantung pada kualitas data historis dan asumsi model. "
                "Gunakan sebagai bahan pertimbangan, bukan angka pasti."
            )
        except Exception as e:
            st.error("Terjadi kesalahan saat melakukan prediksi.")
            st.exception(e)

    st.markdown("---")
    st.write("Developed untuk keperluan skripsi *Prediksi Produksi Tomat di Jawa Barat*.")


if __name__ == "__main__":
    main()
