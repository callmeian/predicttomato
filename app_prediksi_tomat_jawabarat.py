import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
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
    df_panen = pd.read_csv("luas_panen_tomat.csv")
    df_areal = pd.read_csv("luas_areal_tanam_tomat.csv")
    df_produksi = pd.read_csv("produksi_tomat.csv")
    df_cuaca = pd.read_csv("curah_hujan_kelembapan_jawa_barat.csv")

    # Merge sesuai notebook
    df = df_areal.merge(
        df_panen[['kode_provinsi', 'kode_kabupaten_kota', 'tahun', 'luas_panen']],
        on=['kode_provinsi', 'kode_kabupaten_kota', 'tahun'],
        how='inner'
    )

    df = df.merge(
        df_produksi[['kode_provinsi', 'kode_kabupaten_kota', 'tahun', 'produksi_tomat']],
        on=['kode_provinsi', 'kode_kabupaten_kota', 'tahun'],
        how='inner'
    )

    df = df.merge(
        df_cuaca[['kode_provinsi', 'kode_kabupaten_kota', 'tahun',
                   'curah_hujan', 'kelembapan']],
        on=['kode_provinsi', 'kode_kabupaten_kota', 'tahun'],
        how='left'
    )

    df = df.drop_duplicates()
    df = df.dropna(subset=[
        'luas_areal_tanam', 'luas_panen',
        'curah_hujan', 'kelembapan', 'produksi_tomat'
    ])

    return df

# =========================
# Fungsi Latih Model
# =========================
@st.cache_resource
def train_best_model(df: pd.DataFrame):

    num_cols = [
        'luas_areal_tanam',
        'luas_panen',
        'curah_hujan',
        'kelembapan',
        'produksi_tomat'
    ]

    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~(
        (df[num_cols] < (Q1 - 1.5 * IQR)) |
        (df[num_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    df_clean = df.loc[mask].copy()

    X = df_clean[
        [
            'nama_kabupaten_kota',
            'luas_areal_tanam',
            'curah_hujan',
            'kelembapan',
            'luas_panen',
            'tahun'
        ]
    ]
    y = df_clean['produksi_tomat']

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(
        y.values.reshape(-1, 1)
    ).ravel()

    num_features = [
        'luas_areal_tanam',
        'curah_hujan',
        'kelembapan',
        'luas_panen',
        'tahun'
    ]
    cat_features = ['nama_kabupaten_kota']

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_scaled,
        test_size=0.2,
        random_state=42
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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        if name in ["SVR", "Random Forest"]:
            cv_scores = cross_val_score(
                pipe,
                X,
                y_scaled,
                cv=kf,
                scoring="r2"
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = np.nan
            cv_std = np.nan

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2_CV_mean": cv_mean,
            "R2_CV_std": cv_std,
        })

        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(results)
    best_idx = results_df["R2"].idxmax()
    best_model_name = results_df.loc[best_idx, "Model"]

    return trained_pipelines[best_model_name], y_scaler, results_df, best_model_name

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
        "luas panen, curah hujan, kelembapan, dan tahun di setiap kabupaten/kota Jawa Barat."
    )

    df = load_data()

    with st.spinner("Melatih model terbaik berdasarkan data historis..."):
        model, y_scaler, results_df, best_model_name = train_best_model(df)

    st.success(f"Model terbaik yang digunakan **{best_model_name}**")
    with st.expander("Lihat ringkasan performa semua model"):
        st.dataframe(results_df)

    st.markdown("## 📂 Prediksi dari File (CSV / Excel)")
    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # =========================
            # BACA FILE
            # =========================
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            st.subheader("📄 Data yang diupload")
            st.dataframe(df_input.head())

            # =========================
            # VALIDASI KOLOM WAJIB
            # =========================
            required_cols = [
                'nama_kabupaten_kota',
                'luas_areal_tanam',
                'curah_hujan',
                'kelembapan',
                'luas_panen',
                'tahun'
            ]

            missing = [c for c in required_cols if c not in df_input.columns]
            if missing:
                st.error(
                    "Kolom berikut WAJIB ada di file:\n"
                    + ", ".join(missing)
                )
                st.stop()

            # =========================
            # VALIDASI NILAI (MIN–MAX DATASET)
            # =========================
            errors = []

            def cek_rentang(col, min_val, max_val):
                if ((df_input[col] < min_val) | (df_input[col] > max_val)).any():
                    errors.append(
                        f"Kolom '{col}' memiliki nilai di luar rentang "
                        f"{min_val} – {max_val}"
                    )

            cek_rentang('luas_areal_tanam', areal_min, areal_max)
            cek_rentang('luas_panen', panen_min, panen_max)
            cek_rentang('curah_hujan', hujan_min, hujan_max)
            cek_rentang('kelembapan', lembap_min, lembap_max)

            if errors:
                st.warning("⚠️ Prediksi dibatalkan:")
                for e in errors:
                    st.write("❌", e)
                st.stop()

            # =========================
            # CEK KABUPATEN TIDAK DIKENAL
            # =========================
            known_kab = set(
                model.named_steps["preprocessor"]
                    .transformers_[1][1]
                    .categories_[0]
            )

            unknown_kab = set(df_input['nama_kabupaten_kota']) - known_kab

            if unknown_kab:
                st.warning(
                    "⚠️ Beberapa kabupaten tidak ada pada data training:\n"
                    + ", ".join(unknown_kab)
                    + "\n\nModel tetap melakukan prediksi, "
                    "namun akurasi dapat menurun."
                )

            # =========================
            # PREDIKSI
            # =========================
            X_input = df_input[required_cols]
            preds_scaled = model.predict(X_input)

            preds = y_scaler.inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).ravel()

            df_input["prediksi_produksi_tomat"] = preds

            st.success("✅ Prediksi berhasil")
            st.dataframe(df_input)

        except Exception as e:
            st.error("Terjadi kesalahan saat memproses file")
            st.exception(e)


    # =========================
    # Sidebar - Input Pengguna
    # =========================
    st.sidebar.header("Input Fitur Prediksi")

    kabupaten_list = sorted(df["nama_kabupaten_kota"].dropna().unique().tolist())
    nama_kabupaten = st.sidebar.selectbox(
        "Pilih Kabupaten/Kota",
        options=kabupaten_list,
        index=0
    )

    # ===== LIMIT DATASET  =====
    areal_min  = int(df["luas_areal_tanam"].min())
    areal_max  = int(df["luas_areal_tanam"].max())
    panen_min  = int(df["luas_panen"].min())
    panen_max  = int(df["luas_panen"].max())
    hujan_min  = float(df["curah_hujan"].min())
    hujan_max  = float(df["curah_hujan"].max())
    lembap_min = float(df["kelembapan"].min())
    lembap_max = float(df["kelembapan"].max())

    st.sidebar.caption("🔽 Batas berdasarkan dataset")
    st.sidebar.caption(f"- Luas Areal Tanam: {areal_min} – {areal_max} ha")
    st.sidebar.caption(f"- Luas Panen: {panen_min} – {panen_max} ha")
    st.sidebar.caption(f"- Curah Hujan: {hujan_min:.1f} – {hujan_max:.1f} mm")
    st.sidebar.caption(f"- Kelembapan: {lembap_min:.1f} – {lembap_max:.1f} %")

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

    curah_hujan = st.sidebar.number_input(
    "Curah Hujan (mm)",
    min_value=float(hujan_min),
    max_value=float(hujan_max * 1.5),
    value=float(df["curah_hujan"].median()),
    step=1.0
    )

    kelembapan = st.sidebar.number_input(
    "Kelembapan (%)",
    min_value=float(lembap_min),
    max_value=float(lembap_max * 1.2),
    value=float(df["kelembapan"].median()),
    step=1.0
    )

    tahun_list = sorted(df["tahun"].unique().tolist())
    tahun = st.sidebar.selectbox(
        "Pilih Tahun",
        options=tahun_list,
        index=len(tahun_list) - 1
    )

    st.sidebar.markdown("---")
    prediksi_masa_depan = st.sidebar.checkbox("🔮 Prediksi Tahun Masa Depan")

    if prediksi_masa_depan:
        tahun_prediksi = st.sidebar.number_input(
            "Masukkan Tahun Prediksi",
            min_value=int(df["tahun"].max()) + 1,
            max_value=2100,
            value=2025,
            step=1
        )
    else:
        tahun_prediksi = tahun


    predict_button = st.sidebar.button("🔮 Prediksi Produksi Tomat")

    with st.expander("Lihat Data Ringkas"):
        st.write("Contoh 10 baris data gabungan:")
        st.dataframe(df.head(10))
        st.write("Statistik deskriptif fitur numerik:")
        st.dataframe(
            df[["luas_areal_tanam", "luas_panen", "curah_hujan","kelembapan","produksi_tomat"]].describe()
        )

    # =========================
    # Prediksi
    # =========================
    if predict_button:
    
        errors = []

        if luas_areal_tanam < areal_min or luas_areal_tanam > areal_max:
            errors.append(
                f"Luas areal tanam harus di antara {areal_min} – {areal_max} ha"
        )

        if luas_panen < panen_min or luas_panen > panen_max:
            errors.append(
                f"Luas panen harus di antara {panen_min} – {panen_max} ha"
        )

        if curah_hujan < hujan_min or curah_hujan > hujan_max:
            errors.append(
                f"Curah hujan harus di antara {hujan_min:.1f} – {hujan_max:.1f} mm"
        )

        if kelembapan < lembap_min or kelembapan > lembap_max:
            errors.append(
            f"Kelembapan harus di antara {lembap_min:.1f} – {lembap_max:.1f} %"
        )

        # ⛔ JIKA ADA SATU SAJA YANG SALAH
        if errors:
            st.warning("⚠️ Prediksi dibatalkan karena input berikut tidak valid:")
            for err in errors:
                st.write("❌", err)
            st.stop()   # STOP total, tidak lanjut prediksi

        input_data = pd.DataFrame([{
            "nama_kabupaten_kota": nama_kabupaten,
            "luas_areal_tanam": luas_areal_tanam,
            "curah_hujan": curah_hujan,
            "kelembapan": kelembapan,
            "luas_panen": luas_panen,
            "tahun": tahun
        }])
        
        try:
            pred_scaled = model.predict(input_data)

            pred_value = y_scaler.inverse_transform(
                np.array(pred_scaled).reshape(-1, 1)
            )[0][0]

            st.success(f"Perkiraan produksi tomat tahun {tahun_prediksi}: **{pred_value:,.2f} ton**")
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