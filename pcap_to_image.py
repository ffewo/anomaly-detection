import math
import os
from typing import List, Optional, Tuple

import numpy as np
from scapy.all import PcapReader      # type: ignore
from PIL import Image


def read_packet_bytes(
    pcap_path: str,
    max_packets: Optional[int] = None
) -> List[bytes]:
    """
    Verilen PCAP dosyasından ham paket baytlarını okur.
    Her paket, PCAP içindeki tam frame (link layer + IP + TCP/UDP + payload) olarak alınır.

    :param pcap_path: PCAP dosya yolu
    :param max_packets: İstersen üst sınır (None ise hepsi)
    :return: bytes listesi [pkt_bytes_0, pkt_bytes_1, ...]
    """
    packets = []
    with PcapReader(pcap_path) as pcap:
        for i, pkt in enumerate(pcap):
            try:
                packets.append(bytes(pkt.original))
            except AttributeError:
                # bazı scapy sürümlerinde .original yoksa fallback:
                packets.append(bytes(pkt))
            if max_packets is not None and len(packets) >= max_packets:
                break
    return packets


def choose_image_side(
    packet_bytes: List[bytes],
    stat: str = "max"
) -> int:
    """
    Makaledeki fikre uygun olarak, paket uzunluk dağılımına bakıp
    kare görüntü kenar uzunluğunu (N) seçer.

    :param packet_bytes: Paket bayt listesi
    :param stat: "max" | "mean" | "median"
                 - "max": max paket uzunluğuna göre (bilgiyi kaybetmeme açısından önerilen)
    :return: N (görüntü kenar uzunluğu), görüntü boyutu N×N olur.
    """
    lengths = np.array([len(b) for b in packet_bytes], dtype=np.int64)

    if stat == "max":
        base = int(lengths.max())
    elif stat == "mean":
        base = int(lengths.mean())
    elif stat == "median":
        base = int(np.median(lengths))
    else:
        raise ValueError(f"Unknown stat: {stat}")

    # Kare görüntü için N seçimi: N^2 >= base olacak şekilde en küçük N
    side = int(math.ceil(math.sqrt(base)))
    return side


def bytes_to_image_array(
    byte_seq: bytes,
    side: int,
    pad_value: int = 0
) -> np.ndarray:
    """
    Tek bir paket bayt dizisini N×N grayscale görüntü matrisine çevirir.

    - Eğer paket uzunluğu > N^2 ise: ilk N^2 bayt alınır (kırpılır).
    - Eğer paket uzunluğu < N^2 ise: kalan piksel değerleri pad_value ile doldurulur.

    :param byte_seq: Ham paket baytları
    :param side: Görüntü kenar uzunluğu (N)
    :param pad_value: Eksik kalan piksel için doldurma değeri (0 önerilir)
    :return: (N, N) uint8 NumPy array
    """
    total_pixels = side * side

    arr = np.frombuffer(byte_seq[:total_pixels], dtype=np.uint8)

    if arr.size < total_pixels:
        pad = np.full(total_pixels - arr.size, pad_value, dtype=np.uint8)
        arr = np.concatenate([arr, pad])

    img = arr.reshape(side, side)
    return img


def pcap_to_image_arrays(
    pcap_path: str,
    max_packets: Optional[int] = None,
    stat: str = "max",
    side: Optional[int] = None,
    pad_value: int = 0
) -> Tuple[np.ndarray, int]:
    """
    PCAP -> [N×N] görüntü matrisleri

    :param pcap_path: PCAP dosya yolu
    :param max_packets: İstersen üst sınır
    :param stat: side None ise N seçiminde kullanılacak istatistik ("max", "mean", "median")
    :param side: Eğer None değilse, doğrudan bu N kullanılır (örn. 32, 64, 128 ...)
    :param pad_value: Pad değeri (default 0)
    :return: (X, N)  X.shape = (num_packets, N, N), dtype=uint8
    """
    packets = read_packet_bytes(pcap_path, max_packets=max_packets)

    if len(packets) == 0:
        raise RuntimeError("PCAP içinde hiç paket yok.")

    if side is None:
        side = choose_image_side(packets, stat=stat)
        print(f"[INFO] Seçilen görüntü boyutu: {side}×{side}")

    images = [bytes_to_image_array(pkt, side=side, pad_value=pad_value)
              for pkt in packets]

    X = np.stack(images, axis=0)  # (num_packets, side, side)
    return X, side


def save_images_to_folder(
    X: np.ndarray,
    side: int,
    out_dir: str,
    prefix: str = "pkt"
) -> None:
    """
    Üretilen (num_packets, N, N) matrisleri PNG olarak diske kaydeder.

    :param X: (num_packets, N, N) uint8 array
    :param side: N (sadece loglama için)
    :param out_dir: Kayıt klasörü
    :param prefix: Dosya adı prefix'i (örn. pkt_000001.png)
    """
    os.makedirs(out_dir, exist_ok=True)

    num_packets = X.shape[0]
    for i in range(num_packets):
        img_arr = X[i]
        img = Image.fromarray(img_arr, mode="L")  # "L" = 8-bit grayscale
        fname = f"{prefix}_{i:06d}.png"
        img.save(os.path.join(out_dir, fname))

    print(f"[INFO] {num_packets} adet {side}×{side} görüntü {out_dir} klasörüne kaydedildi.")


if __name__ == "__main__":
    # Basit kullanım örneği:
    pcap_path = "IP-Based/Normal/IP-Based Normal Capture.pcap"
    out_dir = "pcap_images/Normal"

    # 1) PCAP -> N×N görüntü matrisleri
    X, side = pcap_to_image_arrays(
        pcap_path,
        max_packets=1000,   # None dersen hepsini işler
        stat="max",         # image size seçimi için "max" istatistiği
        side=None,          # Eğer sabit 32 istiyorsan buraya side=32 yaz
        pad_value=0
    )

    print("X shape:", X.shape)  # (num_packets, N, N)

    # 2) İstersen PNG olarak diske kaydet
    save_images_to_folder(X, side=side, out_dir=out_dir, prefix="pkt")
