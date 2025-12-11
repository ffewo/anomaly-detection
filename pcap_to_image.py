import os
import math
import argparse
from typing import List, Optional

import numpy as np
from scapy.all import PcapReader  # type: ignore
from PIL import Image


# -------------------------------------------------------------------------
# 1) PCAP'ten ham paket baytlarını oku
# -------------------------------------------------------------------------
def read_packet_bytes(
    pcap_path: str,
    max_packets: Optional[int] = None,
) -> List[bytes]:
    """
    Verilen PCAP dosyasından ham paket baytlarını okur.
    Her paket, PCAP içindeki tam frame (link layer + IP + TCP/UDP + payload) olarak alınır.
    """
    packets: List[bytes] = []

    with PcapReader(pcap_path) as pcap:
        for i, pkt in enumerate(pcap):
            if max_packets is not None and i >= max_packets:
                break
            try:
                packets.append(bytes(pkt))
            except Exception:
                # Bozuk paket vs. varsa atla
                continue

    return packets


# -------------------------------------------------------------------------
# 2) window_size paketlik pencereler -> tek byte dizisi
# -------------------------------------------------------------------------
def window_packet_bytes(
    packets: List[bytes],
    window_size: int = 32,
    stride: Optional[int] = None,
) -> List[bytes]:
    """
    Paket listesini window_size paketlik pencerelere böler.
    Her pencere için paketler concat edilip tek byte dizisi döner.

    Örn:
        window_size = 32, stride = 32  -> çakışmasız bloklar
        window_size = 32, stride = 16  -> %50 overlap
        window_size = 32, stride = 1   -> sliding window
    """
    if stride is None:
        stride = window_size

    n = len(packets)
    if n < window_size:
        return []

    windows: List[bytes] = []
    last_start = n - window_size
    for start in range(0, last_start + 1, stride):
        chunk = packets[start:start + window_size]
        concat_bytes = b"".join(chunk)
        windows.append(concat_bytes)

    return windows


# -------------------------------------------------------------------------
# 3) Byte dizisinden sabit boyutlu görüntü matrisi (side x side)
# -------------------------------------------------------------------------
def bytes_to_fixed_image_array(
    data: bytes,
    side: int = 32,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Verilen byte dizisini (len(data)) sabit side x side boyutlu
    uint8 bir görüntü matrisine çevirir.

    - Eğer data, side*side byte'tan UZUNSA -> ilk side*side byte alınır (truncate).
    - Eğer data KISAsa              -> kalan kısım pad_value ile doldurulur.
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
# 4) PCAP -> (num_windows, side, side) tensör
# -------------------------------------------------------------------------
def pcap_to_window_image_arrays(
    pcap_path: str,
    max_packets: Optional[int] = None,
    window_size: int = 32,
    stride: Optional[int] = 16,   # <-- VARSAYILAN STRIDE = 16
    side: int = 32,
    pad_value: int = 0,
) -> np.ndarray:
    """
    PCAP dosyasını okuyup:
        - Paketleri alır
        - window_size paketlik pencerelere böler (stride varsayılan 16)
        - Her pencereyi tek byte dizisine concat eder
        - Her pencereyi side x side görüntüye çevirir

    Dönüş:
        X: np.ndarray, shape = (num_windows, side, side)
    """
    packets = read_packet_bytes(pcap_path, max_packets=max_packets)
    if not packets:
        raise ValueError("PCAP dosyasından paket okunamadı.")

    window_bytes_list = window_packet_bytes(
        packets,
        window_size=window_size,
        stride=stride,
    )
    if not window_bytes_list:
        raise ValueError(
            f"Yeterli paket yok. Toplam paket: {len(packets)}, "
            f"window_size: {window_size}"
        )

    num_windows = len(window_bytes_list)
    X = np.empty((num_windows, side, side), dtype=np.uint8)

    for i, wbytes in enumerate(window_bytes_list):
        X[i] = bytes_to_fixed_image_array(
            wbytes,
            side=side,
            pad_value=pad_value,
        )

    return X


# -------------------------------------------------------------------------
# 5) Görüntüleri diske kaydet
# -------------------------------------------------------------------------
def save_images_to_folder(
    X: np.ndarray,
    out_dir: str,
    prefix: str = "win",
) -> None:
    """
    X: (num_images, H, W) uint8 tensörü
    Her görüntüyü grayscale PNG olarak kaydeder.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_images = X.shape[0]

    for idx in range(num_images):
        img_arr = X[idx]
        img = Image.fromarray(img_arr, mode="L")
        filename = f"{prefix}_{idx:06d}.png"
        img.save(os.path.join(out_dir, filename))


# -------------------------------------------------------------------------
# 6) Komut satırı arayüzü
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PCAP'ten 32 paketlik pencerelerden görüntü üretme script'i (stride=16 varsayılan)"
    )
    parser.add_argument("pcap_path", type=str, help="Girdi PCAP dosyası")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="pcap_windows_32pkt",
        help="Çıktı klasörü (PNG'ler buraya kaydedilir)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=32,
        help="Her görüntüde kullanılacak paket sayısı (varsayılan 32)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,   # <-- KOMUT SATIRINDA DA VARSAYILAN 16
        help="Pencere adımı (varsayılan 16)",
    )
    parser.add_argument(
        "--side",
        type=int,
        default=32,
        help="Görüntü boyutu (side x side, varsayılan 32x32)",
    )
    parser.add_argument(
        "--max_packets",
        type=int,
        default=None,
        help="İşlenecek maksimum paket sayısı (None -> hepsi)",
    )
    args = parser.parse_args()

    X = pcap_to_window_image_arrays(
        pcap_path=args.pcap_path,
        max_packets=args.max_packets,
        window_size=args.window_size,
        stride=args.stride,   # burada 16 geliyor (override etmezsen)
        side=args.side,
        pad_value=0,
    )

    print(f"Üretilen pencere/görüntü sayısı: {X.shape[0]}")
    print(f"Görüntü boyutu: {X.shape[1]} x {X.shape[2]}")

    save_images_to_folder(X, out_dir=args.out_dir, prefix="win")
    print(f"Görüntüler '{args.out_dir}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
