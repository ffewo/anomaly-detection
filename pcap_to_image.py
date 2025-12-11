import os
import argparse
from typing import List, Optional

import numpy as np
from scapy.all import PcapReader  # type: ignore
from PIL import Image

# Tqdm varsa kullan, yoksa boş geç
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


# -------------------------------------------------------------------------
# 1) PCAP Okuma
# -------------------------------------------------------------------------
def read_packet_bytes(pcap_path: str, max_packets: Optional[int] = None) -> List[bytes]:
    packets: List[bytes] = []
    print(f"PCAP okunuyor: {pcap_path} ...")
    with PcapReader(pcap_path) as pcap:
        for i, pkt in enumerate(tqdm(pcap, desc="Paketler Yükleniyor")):
            if max_packets is not None and i >= max_packets:
                break
            try:
                packets.append(bytes(pkt))
            except Exception:
                continue
    return packets


# -------------------------------------------------------------------------
# 2) Windowing (LeNet için optimize edilmiş Trimming)
# -------------------------------------------------------------------------
def window_packet_bytes(
    packets: List[bytes],
    window_size: int,
    stride: int,
    bytes_per_packet: int,
) -> List[bytes]:
    """
    Paketleri pencerelere böler ve hesaplanan limite göre kırpar.
    """
    n = len(packets)
    if n < window_size:
        return []

    windows: List[bytes] = []
    # stride yoksa window_size kadar kaydır
    actual_stride = stride if stride is not None else window_size
    last_start = n - window_size
    
    for start in range(0, last_start + 1, actual_stride):
        chunk = packets[start:start + window_size]
        
        # LeNet için kritik nokta:
        # Her paketin sadece hesaplanan 'bytes_per_packet' kadarını alıyoruz.
        trimmed_chunk = [pkt[:bytes_per_packet] for pkt in chunk]
        
        concat_bytes = b"".join(trimmed_chunk)
        windows.append(concat_bytes)

    return windows


# -------------------------------------------------------------------------
# 3) Byte -> Image (32x32 Sabit)
# -------------------------------------------------------------------------
def bytes_to_lenet_image(
    data: bytes,
    side: int = 32,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Veriyi 32x32 (veya belirtilen side) matrise dönüştürür.
    """
    total = side * side
    arr = np.frombuffer(data, dtype=np.uint8)
    
    if arr.size >= total:
        arr = arr[:total]
    else:
        tmp = np.full(total, pad_value, dtype=np.uint8)
        tmp[:arr.size] = arr
        arr = tmp

    img = arr.reshape(side, side)
    return img


# -------------------------------------------------------------------------
# 4) Ana Akış
# -------------------------------------------------------------------------
def pcap_to_lenet_ready_arrays(
    pcap_path: str,
    max_packets: Optional[int] = None,
    window_size: int = 16,  # <-- LENET İÇİN VARSAYILAN 16 (Daha fazla paket sığmaz)
    stride: Optional[int] = 8,
    side: int = 32,         # <-- LENET STANDARD (32x32)
) -> np.ndarray:
    
    # 1. Paketleri Oku
    packets = read_packet_bytes(pcap_path, max_packets=max_packets)
    if not packets:
        raise ValueError("PCAP boş veya okunamadı.")

    # 2. LeNet Kısıt Kontrolü ve Limit Hesabı
    total_capacity = side * side  # 1024
    calculated_limit = total_capacity // window_size
    
    print(f"\n--- LeNet Uyumluluk Kontrolü ---")
    print(f"Giriş Boyutu    : {side}x{side} ({total_capacity} byte)")
    print(f"Pencere Boyutu  : {window_size} paket")
    print(f"Paket Başına    : {calculated_limit} byte ayrıldı.")

    if calculated_limit < 54:
        print("UYARI: Paket başına ayrılan alan < 54 byte.")
        print("       TCP/UDP başlıkları ve port bilgileri kesilebilir!")
        print("       ÖNERİ: --window_size parametresini düşürün (örn: 12 veya 16).")
    else:
        print("DURUM: İdeal. Headerlar ve payload başlangıcı sığıyor.")
    print("--------------------------------\n")

    # 3. Pencereleri Oluştur
    window_bytes_list = window_packet_bytes(
        packets,
        window_size=window_size,
        stride=stride,
        bytes_per_packet=calculated_limit
    )

    if not window_bytes_list:
        raise ValueError(f"Yeterli paket yok. ({len(packets)} < {window_size})")

    # 4. Görüntüye Çevir
    num_windows = len(window_bytes_list)
    X = np.empty((num_windows, side, side), dtype=np.uint8)

    for i, wbytes in enumerate(window_bytes_list):
        X[i] = bytes_to_lenet_image(wbytes, side=side)

    return X


def save_images(X: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Kayıt başladı: {out_dir}")
    for idx in range(X.shape[0]):
        img = Image.fromarray(X[idx], mode="L")
        img.save(os.path.join(out_dir, f"image_{idx:06d}.png"))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PCAP -> LeNet (32x32) Dataset Generator")
    parser.add_argument("pcap_path", type=str)
    parser.add_argument("--out_dir", type=str, default="pcap_images")
    
    # LeNet için kritik varsayılanlar
    parser.add_argument("--side", type=int, default=32, help="LeNet için 32 olmalı")
    
    # Window size 16 seçildi çünkü: 1024 / 16 = 64 byte. 
    # Bu da TCP headerlarını kurtarmak için gereken minimum alandır.
    parser.add_argument("--window_size", type=int, default=16, help="LeNet için 16 önerilir")
    
    parser.add_argument("--stride", type=int, default=8, help="Veri çoğaltmak için düşük tutulabilir")
    parser.add_argument("--max_packets", type=int, default=None)

    args = parser.parse_args()

    X = pcap_to_lenet_ready_arrays(
        pcap_path=args.pcap_path,
        max_packets=args.max_packets,
        window_size=args.window_size,
        stride=args.stride,
        side=args.side
    )

    print(f"Üretilen Tensör: {X.shape}")
    save_images(X, args.out_dir)

if __name__ == "__main__":
    main()