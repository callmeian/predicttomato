import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================
# Fungsi Load Data
# =========================

@st.cache_data
def load_data():
    """Memuat dan menggabungkan 3 dataset tomat."""
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

    # Hapus duplikat & NaN penting (opsional)
    df = df.drop_duplicates()
    df = df.dropna(subset=["luas_areal_tanam", "luas_panen", "produksi_tomat"])

    return df


# =========================
# Fungsi Latih Model di Dalam App
# =========================

@st.cache_resource
def train_best_model(df: pd.DataFrame):
    """
    Melatih 3 model (LR, SVR, RF) dan mengembalikan pipeline terbaik.
    Dilatih ulang setiap ada perubahan data (cache_resource menyimpan hasilnya).
    """
    X = df[["nama_kabupaten_kota", "luas_areal_tanam", "luas_panen", "tahun"]]
    y = df["produksi_tomat"]

    num_features = ["luas_areal_tanam", "luas_panen", "tahun"]
    cat_features = ["nama_kabupaten_kota"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel="linear"),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # K-Fold hanya untuk SVR & RF (sesuai metodologi)
        if name in ["SVR", "Random Forest"]:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = np.nan
            cv_std = np.nan

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "Model": name,
            "R2": r2,
            "MSE": mse,
            "MAE": mae,
            "R2_CV_mean": cv_mean,
            "R2_CV_std": cv_std,
        })

        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(results)
    best_idx = results_df["R2"].idxmax()
    best_model_name = results_df.loc[best_idx, "Model"]
    best_pipeline = trained_pipelines[best_model_name]

    return best_pipeline, results_df, best_model_name


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

    # Load data
    df = load_data()

    with st.spinner("Melatih model terbaik berdasarkan data historis..."):
        model, results_df, best_model_name = train_best_model(df)

    st.success(f"Model terbaik yang digunakan **{best_model_name}**")
    with st.expander("Lihat ringkasan performa semua model"):
        st.dataframe(results_df)

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

    # LIMIT DATASET
    areal_min  = int(df["luas_areal_tanam"].min())
    areal_max  = int(df["luas_areal_tanam"].max())
    panen_min  = int(df["luas_panen"].min())
    panen_max  = int(df["luas_panen"].max())

    # Info MIN MAX
    st.sidebar.caption(f"🔽 Batas berdasarkan dataset")
    st.sidebar.caption(f"- Luas Areal Tanam: {areal_min} – {areal_max} ha")
    st.sidebar.caption(f"- Luas Panen: {panen_min} – {panen_max} ha")

    # Input integer
    luas_areal_tanam = st.sidebar.number_input(
    "Luas Areal Tanam (ha)",
    min_value=areal_min,
    max_value=int(areal_max * 1.5),
    value=int(df["luas_areal_tanam"].median()),
    step=1
    )

    luas_panen = st.sidebar.number_input(
    "Luas Panen (ha)",
    min_value=panen_min,
    max_value=int(panen_max * 1.5),
    value=int(df["luas_panen"].median()),
    step=1
    )

    tahun_list = sorted(df["tahun"].unique().tolist())
    tahun = st.sidebar.selectbox(
    "Pilih Tahun",
    options=tahun_list,
    index=len(tahun_list)-1
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
            st.success(f"Perkiraan produksi tomat: **{pred_value:,.2f} ton**")

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