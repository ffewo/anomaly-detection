# PCAP to Image & DROCC Anomaly Detection

Bu proje, ağ paketlerini (PCAP dosyaları) görüntülere dönüştürüp DROCC (Deep Robust One-Class Classification) algoritması ile anomali tespiti yapan bir makine öğrenmesi sistemidir.

## Özellikler

- PCAP dosyalarını grayscale görüntülere dönüştürme
- LeNet-5 tabanlı DROCC modeli ile one-class learning
- Anomali tespiti için adversarial training

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. PCAP Dosyalarını Görüntülere Dönüştürme

```bash
python pcap_to_image.py
```

Bu script, PCAP dosyalarını PNG görüntülerine dönüştürür ve `pcap_images/` klasörüne kaydeder.

### 2. Model Eğitimi

```bash
python train_drocc_packets.py
```

Model sadece normal paketlerle eğitilir ve anomali tespiti için kullanılır.

## Klasör Yapısı

```
.
├── pcap_to_image.py          # PCAP -> Image dönüştürücü
├── train_drocc_packets.py    # DROCC model eğitimi
├── requirements.txt          # Python bağımlılıkları
├── pcap_images/              # Üretilen görüntüler
│   ├── Normal/              # Normal paket görüntüleri
│   └── Attack/              # Saldırı paket görüntüleri (test için)
└── IP-Based/                # PCAP dosyaları
    ├── Normal/
    └── Malicious/
```

## Gereksinimler

- Python 3.7+
- PyTorch
- NumPy
- Scapy
- Pillow (PIL)
- torchvision

## Lisans

MIT

