import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from PIL import Image


# ================== AYARLAR ==================

CSV_PATH = "wustl-ehms-2020_with_attacks_categories.csv"   # CSV dosyanın yolu
LABEL_COL = "Label"              # Etiket sütununun adı
NORMAL_LABEL_VALUE = 0           # 0 = normal, 1 = saldırı

OUTPUT_DIR_TRAIN_NORMAL = "data/train/normal"   # DROCC-LeNet eğitimi için normal imgeler
OUTPUT_DIR_TEST_ATTACK = "data/test/attack"     # Test için saldırı imgeleri

N_PCA_COMPONENTS = 32          # Makaledeki gibi 32 bileşen
WINDOW_SIZE = 32               # 32 ardışık satır → 32x32 görüntü
STRIDE = 1                     # Kaydırma adımı (1 = her satırda yeni pencere)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 1               # GRAYSCALE görüntü üretmek için 1 kanal


# ================== YARDIMCI FONKSİYONLAR ==================


def load_and_split_by_label(csv_path: str, label_col: str, normal_label_value):
    """CSV yükle, normal ve saldırı satırlarını ayır."""
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {df.columns.tolist()}")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # 0 → normal, 1 → saldırı
    mask_normal = (y == normal_label_value)
    X_normal = X[mask_normal].reset_index(drop=True)
    X_attack = X[~mask_normal].reset_index(drop=True)
    y_attack = y[~mask_normal].reset_index(drop=True)

    if X_normal.empty:
        raise ValueError(f"Normal label '{normal_label_value}' ile eşleşen hiç satır bulunamadı.")

    print(f"Normal satır sayısı: {len(X_normal)}")
    print(f"Saldırı satır sayısı: {len(X_attack)}")

    return X_normal, X_attack, y_attack


def build_preprocess_pipeline(X_train_normal: pd.DataFrame):
    """
    One-hot encoding + SimpleImputer + QuantileTransformer pipeline'ı.
    - Kategorik: NaN → en sık görülen, sonra OHE
    - Sayısal: NaN → medyan
    - Sonra hepsi için QuantileTransformer
    """
    cat_cols = X_train_normal.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train_normal.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(
            ("cat", cat_pipeline, cat_cols)
        )

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(
            ("num", num_pipeline, num_cols)
        )

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    qt = QuantileTransformer(output_distribution="uniform")

    pipeline = Pipeline([
        ("col_tf", col_transformer),
        ("qt", qt),
    ])

    return pipeline


def apply_pca(X_norm: np.ndarray, n_components: int = 32):
    """PCA ile boyut azaltma (normal veri üzerinden fit)."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    return X_pca, pca


def transform_with_fitted_pca(pca: PCA, X_norm: np.ndarray):
    """Fit edilmiş PCA ile başka veri (saldırı) dönüştür."""
    return pca.transform(X_norm)


def scale_for_image_with_reference(X_ref: np.ndarray, X_target: np.ndarray):
    """
    Her PCA bileşeni için ayrı min–max kullanarak 0–255'e ölçekle.
    Böylece her sütun (bileşen) kendi aralığında normalize olur,
    düz gri bloklar azalır.
    """
    mins = X_ref.min(axis=0)      # (n_features,)
    maxs = X_ref.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    X_ref_norm = (X_ref - mins) / ranges
    X_target_norm = (X_target - mins) / ranges

    X_ref_scaled = (X_ref_norm * 255).clip(0, 255).astype(np.uint8)
    X_target_scaled = (X_target_norm * 255).clip(0, 255).astype(np.uint8)

    return X_ref_scaled, X_target_scaled


def create_image_windows(X_pca_scaled: np.ndarray,
                         window_size: int,
                         stride: int,
                         img_height: int,
                         img_width: int,
                         img_channels: int,
                         output_dir: str,
                         labels: pd.Series | None = None):
    """
    PCA sonrası 0–255'e ölçeklenmiş veriyi sliding window ile görüntüye çevirir.
    - Her window: window_size satır → img_height x img_width matris.
    - stride: pencere kaydırma adımı (ör. 1, 4, 8, 32...)
    - labels verilirse (saldırı tarafı) çoğunluk etiketine göre alt klasör açar.
    - labels None ise (normal taraf) tek klasöre yazar.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_samples, n_features = X_pca_scaled.shape

    if n_features != img_width:
        raise ValueError(
            f"PCA feature sayısı ({n_features}) img_width ({img_width}) ile eşleşmiyor. "
            f"img_width'i {n_features} yap ya da PCA bileşen sayısını değiştir."
        )

    img_count = 0

    # ----------- BURADA STRIDE KULLANIYORUZ -----------
    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_data = X_pca_scaled[start_idx:end_idx, :]

        if window_data.shape[0] != img_height:
            continue

        img_array = window_data.reshape(img_height, img_width)

        if img_channels == 1:
            img_2d = img_array
            mode = "L"
        elif img_channels == 3:
            img_3d = np.stack([img_array, img_array, img_array], axis=-1)
            img_2d = img_3d
            mode = "RGB"
        else:
            raise ValueError("img_channels sadece 1 veya 3 olmalı.")

        if labels is not None:
            window_labels = labels.iloc[start_idx:end_idx]
            majority_label = int(window_labels.value_counts().index[0])
            if majority_label == 0:
                class_name = "normal"
            else:
                class_name = "attack"
            class_dir = Path(output_dir) / class_name
        else:
            class_dir = Path(output_dir)

        class_dir.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(img_2d, mode=mode)
        img.save(class_dir / f"img_{img_count:06d}.png")
        img_count += 1

    print(f"'{output_dir}' dizininde {img_count} görüntü üretildi.")


# ================== ANA AKIŞ ==================


def main():
    # 1) CSV yükle, normal & saldırı ayır
    X_normal, X_attack, y_attack = load_and_split_by_label(
        CSV_PATH, LABEL_COL, NORMAL_LABEL_VALUE
    )

    # 2) Normal veri üzerinden preprocess pipeline (Imputer + OHE + Quantile) öğren
    preprocess_pipe = build_preprocess_pipeline(X_normal)

    # Normal ve saldırıyı aynı pipeline ile dönüştür
    X_normal_norm = preprocess_pipe.fit_transform(X_normal)
    X_attack_norm = preprocess_pipe.transform(X_attack)

    # 3) PCA: sadece normal üzerinde fit, saldırıya transform
    X_normal_pca, pca_model = apply_pca(X_normal_norm, n_components=N_PCA_COMPONENTS)
    X_attack_pca = transform_with_fitted_pca(pca_model, X_attack_norm)

    # 4) 0–255 ölçek: min/max'i normalden al, saldırıya aynı ölçeği uygula (sütun bazlı)
    X_normal_scaled, X_attack_scaled = scale_for_image_with_reference(
        X_normal_pca, X_attack_pca
    )

    # 5) Normal veri → eğitim (DROCC) grayscale görüntüleri (stride'lı)
    create_image_windows(
        X_pca_scaled=X_normal_scaled,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_channels=IMG_CHANNELS,
        output_dir=OUTPUT_DIR_TRAIN_NORMAL,
        labels=None
    )

    # 6) Saldırı veri → test grayscale görüntüleri (alt klasör: normal/attack)
    create_image_windows(
        X_pca_scaled=X_attack_scaled,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_channels=IMG_CHANNELS,
        output_dir=OUTPUT_DIR_TEST_ATTACK,
        labels=y_attack
    )


if __name__ == "__main__":
    main()
