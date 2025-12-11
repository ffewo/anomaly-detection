import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from PIL import Image


# ================== VARSAYILAN AYARLAR ==================

DEFAULT_CSV_PATH = "wustl-ehms-2020_with_attacks_categories.csv"
DEFAULT_LABEL_COL = "Label"
DEFAULT_NORMAL_LABEL_VALUE = 0

DEFAULT_OUTPUT_DIR_TRAIN_NORMAL = "data/train/normal"
DEFAULT_OUTPUT_DIR_TEST_NORMAL = "data/test/normal"
DEFAULT_OUTPUT_DIR_TEST_ATTACK = "data/test/attack"

DEFAULT_N_PCA_COMPONENTS = 32
DEFAULT_WINDOW_SIZE = 32
DEFAULT_STRIDE = 16

DEFAULT_IMG_HEIGHT = 32
DEFAULT_IMG_WIDTH = 32
DEFAULT_IMG_CHANNELS = 1  # grayscale

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_RANDOM_STATE = 42


# ================== YARDIMCI FONKSİYONLAR ==================


def load_and_split_by_label(csv_path: str, label_col: str, normal_label_value):
    """CSV yükle, normal ve saldırı satırlarını ayır."""
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {df.columns.tolist()}")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # 0 → normal, Diğerleri → saldırı
    mask_normal = (y == normal_label_value)
    X_normal = X[mask_normal].reset_index(drop=True)
    
    # Normal olmayan her şey saldırı/anomali kabul edilir
    X_attack = X[~mask_normal].reset_index(drop=True)

    if X_normal.empty:
        raise ValueError(f"Normal label '{normal_label_value}' ile eşleşen hiç satır bulunamadı.")

    print(f"Normal satır sayısı: {len(X_normal)}")
    print(f"Saldırı satır sayısı: {len(X_attack)}")

    return X_normal, X_attack


def build_preprocess_pipeline(X_train_normal: pd.DataFrame):
    """
    Pipeline: Imputer -> OHE -> QuantileTransformer (Normal Dağılım)
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

    # ÖNEMLİ: PCA öncesi veriyi Gaussian (Normal) dağılıma çevirmek performansı artırır.
    qt = QuantileTransformer(output_distribution="normal", random_state=42)

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
    """Fit edilmiş PCA ile saldırı verisini dönüştür."""
    return pca.transform(X_norm)


def scale_for_image_with_reference(X_ref: np.ndarray, X_target: np.ndarray):
    """
    Referans veriye (normal) göre 0-255 ölçekleme.
    """
    mins = X_ref.min(axis=0)
    maxs = X_ref.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    X_ref_norm = (X_ref - mins) / ranges
    X_target_norm = (X_target - mins) / ranges

    X_ref_scaled = (X_ref_norm * 255).clip(0, 255).astype(np.uint8)
    X_target_scaled = (X_target_norm * 255).clip(0, 255).astype(np.uint8)

    return X_ref_scaled, X_target_scaled


def _window_to_image(
    window_data: np.ndarray,
    img_height: int,
    img_width: int,
    img_channels: int,
):
    """Tek bir pencereyi görüntü array'ine çevir."""
    if window_data.shape[0] != img_height:
        return None

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

    return img_2d, mode


def create_image_windows_split_normal(
    X_pca_scaled: np.ndarray,
    window_size: int,
    stride: int,
    img_height: int,
    img_width: int,
    img_channels: int,
    output_train_dir: str,
    output_test_dir: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
):
    """Normal veriyi Train/Test olarak ayırıp kaydeder."""
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    n_samples, n_features = X_pca_scaled.shape

    # Bütün pencere başlangıç indexleri
    start_indices = list(range(0, n_samples - window_size + 1, stride))
    if not start_indices:
        print("Normal veri için pencere üretilemedi.")
        return

    rng = np.random.RandomState(random_state)
    rng.shuffle(start_indices)

    n_train = int(len(start_indices) * train_ratio)
    train_starts = start_indices[:n_train]
    test_starts = start_indices[n_train:]

    print(f"Normal pencereler: toplam={len(start_indices)}, "
          f"train={len(train_starts)}, test={len(test_starts)}")

    # Train pencereleri
    train_count = 0
    for start_idx in train_starts:
        end_idx = start_idx + window_size
        window_data = X_pca_scaled[start_idx:end_idx, :]

        result = _window_to_image(window_data, img_height, img_width, img_channels)
        if result is None: continue
        img_2d, mode = result

        img = Image.fromarray(img_2d, mode=mode)
        img.save(Path(output_train_dir) / f"img_{train_count:06d}.png")
        train_count += 1

    # Test-normal pencereleri
    test_count = 0
    for start_idx in test_starts:
        end_idx = start_idx + window_size
        window_data = X_pca_scaled[start_idx:end_idx, :]

        result = _window_to_image(window_data, img_height, img_width, img_channels)
        if result is None: continue
        img_2d, mode = result

        img = Image.fromarray(img_2d, mode=mode)
        img.save(Path(output_test_dir) / f"img_{test_count:06d}.png")
        test_count += 1

    print(f"Train normal görüntü sayısı: {train_count}")
    print(f"Test normal görüntü sayısı: {test_count}")


def create_image_windows_attack(
    X_pca_scaled: np.ndarray,
    window_size: int,
    stride: int,
    img_height: int,
    img_width: int,
    img_channels: int,
    output_dir: str,
):
    """
    SALDIRI veri setinden gelen pencereleri kaydeder.
    Bu veriler load aşamasında filtrelendiği için hepsi saldırı kabul edilir.
    """
    n_samples = X_pca_scaled.shape[0]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_count = 0

    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_data = X_pca_scaled[start_idx:end_idx, :]

        result = _window_to_image(window_data, img_height, img_width, img_channels)
        if result is None: continue
        img_2d, mode = result

        # Dosya ismi sadece sıra numarası içerir
        filename = f"img_{img_count:06d}.png"
        
        img = Image.fromarray(img_2d, mode=mode)
        img.save(out_path / filename)
        img_count += 1

    print(f"'{output_dir}' dizininde {img_count} saldırı testi görüntüsü üretildi.")


# ================== ANA AKIŞ ==================


def main(args: argparse.Namespace):
    # GÜVENLİK KONTROLÜ: Boyut uyuşmazlığını baştan çöz.
    if args.img_width != args.n_components:
        print(f"UYARI: img_width ({args.img_width}) ve n_components ({args.n_components}) eşitleniyor.")
        args.img_width = args.n_components

    # 1) CSV yükle
    X_normal, X_attack = load_and_split_by_label(
        csv_path=args.csv_path,
        label_col=args.label_col,
        normal_label_value=args.normal_label,
    )

    # 2) Preprocess Pipeline (Sadece Normal veride öğrenir)
    preprocess_pipe = build_preprocess_pipeline(X_normal)
    X_normal_norm = preprocess_pipe.fit_transform(X_normal)
    X_attack_norm = preprocess_pipe.transform(X_attack)

    # 3) PCA (Sadece Normal veride öğrenir)
    X_normal_pca, pca_model = apply_pca(X_normal_norm, n_components=args.n_components)
    X_attack_pca = transform_with_fitted_pca(pca_model, X_attack_norm)

    # 4) Scale to 0-255 (Referans: Normal veri)
    X_normal_scaled, X_attack_scaled = scale_for_image_with_reference(
        X_normal_pca, X_attack_pca
    )

    # 5) Normal Veri -> Train/Test
    create_image_windows_split_normal(
        X_pca_scaled=X_normal_scaled,
        window_size=args.window_size,
        stride=args.stride,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
        output_train_dir=args.output_train_dir,
        output_test_dir=args.output_test_normal_dir,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
    )

    # 6) Saldırı Veri -> Test Attack Klasörü
    create_image_windows_attack(
        X_pca_scaled=X_attack_scaled,
        window_size=args.window_size,
        stride=args.stride,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
        output_dir=args.output_test_attack_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV → PCA → Image Converter (Optimized for Anomaly Detection)"
    )

    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--normal-label", type=int, default=DEFAULT_NORMAL_LABEL_VALUE)
    
    parser.add_argument("--output-train-dir", type=str, default=DEFAULT_OUTPUT_DIR_TRAIN_NORMAL)
    parser.add_argument("--output-test-normal-dir", type=str, default=DEFAULT_OUTPUT_DIR_TEST_NORMAL)
    parser.add_argument("--output-test-attack-dir", type=str, default=DEFAULT_OUTPUT_DIR_TEST_ATTACK)
    
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_PCA_COMPONENTS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    
    parser.add_argument("--img-height", type=int, default=DEFAULT_IMG_HEIGHT)
    parser.add_argument("--img-width", type=int, default=DEFAULT_IMG_WIDTH)
    parser.add_argument("--img-channels", type=int, default=DEFAULT_IMG_CHANNELS)
    
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    args = parser.parse_args()
    main(args)