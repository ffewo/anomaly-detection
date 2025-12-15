import os
import argparse
from typing import List, Optional

import numpy as np
from scapy.all import PcapReader 
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

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

def window_packet_bytes(
    packets: List[bytes],
    window_size: int,
    stride: int,
    bytes_per_packet: int,
) -> List[bytes]:

    n = len(packets)
    if n < window_size:
        return []

    windows: List[bytes] = []
    actual_stride = stride if stride is not None else window_size
    last_start = n - window_size
    
    for start in range(0, last_start + 1, actual_stride):
        chunk = packets[start:start + window_size]
        trimmed_chunk = [pkt[:bytes_per_packet] for pkt in chunk]
        concat_bytes = b"".join(trimmed_chunk)
        windows.append(concat_bytes)

    return windows

def bytes_to_lenet_image(
    data: bytes,
    side: int = 32,
    pad_value: int = 0,
) -> np.ndarray:

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

def pcap_to_lenet_ready_arrays(
    pcap_path: str,
    max_packets: Optional[int] = None,
    window_size: int = 16,   
    stride: Optional[int] = 8,
    side: int = 32,         
) -> np.ndarray:
    
    packets = read_packet_bytes(pcap_path, max_packets=max_packets)
    if not packets:
        raise ValueError("PCAP boş veya okunamadı.")

    total_capacity = side * side
    calculated_limit = total_capacity // window_size
    
    print(f"Giriş Boyutu     : {side}x{side} ({total_capacity} byte)")
    print(f"Pencere Boyutu   : {window_size} paket")
    print(f"Paket Başına     : {calculated_limit} byte ayrıldı.")
    print("--------------------------------\n")

    window_bytes_list = window_packet_bytes(
        packets,
        window_size=window_size,
        stride=stride,
        bytes_per_packet=calculated_limit
    )

    if not window_bytes_list:
        raise ValueError(f"Yeterli paket yok. ({len(packets)} < {window_size})")

    num_windows = len(window_bytes_list)
    X = np.empty((num_windows, side, side), dtype=np.uint8)

    for i, wbytes in enumerate(window_bytes_list):
        X[i] = bytes_to_lenet_image(wbytes, side=side)

    return X

def save_images(X: np.ndarray, out_dir: str, prefix: str = "image"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"--> Kayıt yapılıyor: {out_dir} ({len(X)} adet)")
    
    iterator = range(X.shape[0])
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Kaydediliyor ({os.path.basename(out_dir)})")
    except ImportError:
        pass

    for idx in iterator:
        img = Image.fromarray(X[idx], mode="L")
        img.save(os.path.join(out_dir, f"{prefix}_{idx:06d}.png"))

def main():
    parser = argparse.ArgumentParser(description="PCAP -> LeNet Dataset Generator (Normal/Attack Modes)")
    parser.add_argument("pcap_path", type=str)
    
    # Yeni argüman: Veri tipi (normal mi attack mı?)
    parser.add_argument("--dataset_type", type=str, required=True, choices=['normal', 'attack'], 
                        help="'normal' seçilirse %80 eğitim %20 test bölünür. 'attack' seçilirse hepsi tek klasöre gider.")
    
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max_packets", type=int, default=None)

    args = parser.parse_args()

    # 1. Veriyi Hazırla
    X = pcap_to_lenet_ready_arrays(
        pcap_path=args.pcap_path,
        max_packets=args.max_packets,
        window_size=args.window_size,
        stride=args.stride,
        side=args.side
    )
    print(f"Toplam Üretilen Görüntü: {X.shape[0]}")

    # 2. Moduna Göre Kaydetme Mantığı
    if args.dataset_type == 'normal':
        # --- NORMAL MODU (%80 Train - %20 Test) ---
        print("\n[MOD] Normal Veri İşleme (Karıştır ve Böl)")
        
        # Karıştır
        indices = np.random.permutation(len(X))
        X = X[indices]
        
        # Böl
        split_index = int(len(X) * 0.8)
        X_train = X[:split_index]
        X_test = X[split_index:]
        
        train_dir = os.path.join("pcap_images", "Normal")
        test_dir = os.path.join("pcap_images", "test") 

        print(f"Eğitim Seti (%80): {len(X_train)} adet -> {train_dir}")
        print(f"Test Seti   (%20): {len(X_test)} adet -> {test_dir}")
        
        save_images(X_train, train_dir, prefix="train_norm")
        save_images(X_test, test_dir, prefix="test_norm")
        
    elif args.dataset_type == 'attack':
        # --- ATTACK MODU (Hepsi Tek Klasöre) ---
        print("\n[MOD] Saldırı Verisi İşleme (Bölme Yok, Hepsi Kaydediliyor)")
        
        # Hedef: pcap_images/Attacks (Test amaçlı kullanılacağı için)
        # İsterseniz test/attack altına da koyabilirsiniz ama isteğiniz Attacks klasörüydü.
        attack_dir = os.path.join("pcap_images", "Attacks")
        
        print(f"Tüm Veri: {len(X)} adet -> {attack_dir}")
        
        save_images(X, attack_dir, prefix="attack")

    print("\n[TAMAMLANDI] İşlem başarıyla bitti.")

if __name__ == "__main__":
    main()