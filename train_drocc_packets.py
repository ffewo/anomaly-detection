import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # Confusion Matrix görselleştirmesi için (opsiyonel ama önerilir)

# ============================
# 1) Dataset / DataLoader
# ============================

class PacketImageDataset(Dataset):
    """
    Klasördeki PNG (veya JPG) paket görüntülerini okur.
    Tek kanallı (1 x 32 x 32) tensör ve sabit bir label döndürür.
    DROCC için eğitimde label aslında kullanılmıyor (normal = 1).
    """
    def __init__(self, root_dir: str, label: int = 1, img_size: int = 32):
        """
        :param root_dir: Görüntülerin bulunduğu klasör (ör: data/train/normal)
        :param label: Bu klasörün etiketi (normal=1, attack=0 vb.)
        :param img_size: LeNet için hedef boyut (32 önerilir)
        """
        self.root_dir = root_dir
        self.label = label
        self.img_size = img_size

        exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(exts)
        ]
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"{root_dir} içinde hiç görüntü bulunamadı.")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),   # tek kanal
            transforms.Resize((img_size, img_size)),       # 32x32
            transforms.ToTensor(),                         # [0,1], shape: (1,H,W)
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.files[idx]
        img = Image.open(img_path)
        img = self.transform(img)  # (1, img_size, img_size)
        label = self.label
        return img, label


# ============================
# 2) LeNet-5 Backbone + DROCC
# ============================

class LeNet5Backbone(nn.Module):
    """
    LeNet-5'in embedding (F6) katmanına kadar olan kısmı.
    Girdi: (B,1,32,32)
    Çıkış: (B,84)
    """
    def __init__(self):
        super().__init__()
        # C1: 1x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # C3: 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # C5: 16x5x5 -> 120x1x1
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # F6: 120 -> 84
        self.fc1 = nn.Linear(120, 84)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2)                 # 6x14x14
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)                 # 16x5x5
        x = torch.tanh(self.conv3(x))          # 120x1x1
        x = x.view(x.size(0), -1)              # (B,120)
        x = torch.tanh(self.fc1(x))            # (B,84)
        return x                               # embedding z


class DROCCHead(nn.Module):
    """
    Embedding (z) üzerine tek-sınıf / binary classifier.
    DROCC'de pozitif sınıf 'normal', adversarial negatifler 'anomali'.
    """
    def __init__(self, emb_dim: int = 84):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)  # logit

    def forward(self, z):
        logit = self.fc(z)   # (B,1)
        return logit.squeeze(-1)  # (B,)


class LeNet5DROCC(nn.Module):
    """
    Backbone + DROCC head birleşik model.
    """
    def __init__(self):
        super().__init__()
        self.backbone = LeNet5Backbone()
        self.head = DROCCHead(emb_dim=84)

    def forward(self, x):
        z = self.backbone(x)
        logit = self.head(z)
        return z, logit


# ============================
# 3) DROCC Loss Fonksiyonu
# ============================

def project_to_sphere(z, z_adv, R, eps):
    """
    z etrafında R ile (1+eps)*R arasında halka (sphere shell) içine projeksiyon.
    z_adv: başlangıç pertürbe edilmiş embedding
    """
    diff = z_adv - z
    dist = diff.norm(p=2, dim=1, keepdim=True) + 1e-8

    lower = R
    upper = (1.0 + eps) * R

    scale = torch.clamp(dist, min=lower, max=upper) / dist
    z_proj = z + diff * scale
    return z_proj


def estimate_radius(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: str = "cuda", 
    num_batches: int = 10
) -> float:
    """
    Eğitimden önce verinin embedding uzayındaki ortalama normunu hesaplar.
    Bu, R (yarıçap) parametresini başlatmak için kullanılır.
    """
    model.eval()
    norms = []
    
    print(f"[*] Radius (R) tahmini yapılıyor ({num_batches} batch)...")
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            # Modelden z (embedding) alıyoruz
            z, _ = model(x) 
            
            # Her bir örneğin L2 normunu hesapla
            batch_norms = torch.norm(z, p=2, dim=1)
            norms.append(batch_norms)
    
    # Tüm normların ortalamasını al
    all_norms = torch.cat(norms)
    avg_radius = all_norms.mean().item()
    
    print(f"[*] Tahmin edilen R: {avg_radius:.4f}")
    return avg_radius

def drocc_loss(
    model: LeNet5DROCC,
    x: torch.Tensor,
    R: float = 1.0,
    eps: float = 0.25,
    n_steps: int = 5,
    step_size: float = 0.03,
    lambda_adv: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Sadeleştirilmiş DROCC loss.
    - model: LeNet5DROCC
    - x: sadece 'normal' veriler (one-class)
    """
    device = x.device
    batch_size = x.size(0)

    # 1) Normal embedding ve logit
    z, logit_pos = model(x)
    y_pos = torch.ones(batch_size, device=device)

    # 2) Adversarial negatif embedding üret
    z_adv = z + 0.1 * torch.randn_like(z)
    z_adv = project_to_sphere(z, z_adv, R, eps)
    z_adv.requires_grad_(True)

    for _ in range(n_steps):
        logit_adv = model.head(z_adv)
        y_fake = torch.ones(batch_size, device=device)
        loss_adv = F.binary_cross_entropy_with_logits(logit_adv, y_fake)
        grad = torch.autograd.grad(loss_adv, z_adv)[0]
        z_adv = z_adv + step_size * grad
        z_adv = project_to_sphere(z, z_adv, R, eps)
        z_adv = z_adv.detach()
        z_adv.requires_grad_(True)

    z_adv = z_adv.detach()
    logit_neg = model.head(z_adv)
    y_neg = torch.zeros(batch_size, device=device)

    loss_pos = F.binary_cross_entropy_with_logits(logit_pos, y_pos)
    loss_neg = F.binary_cross_entropy_with_logits(logit_neg, y_neg)

    loss = loss_pos + lambda_adv * loss_neg
    return loss, loss_pos.item(), loss_neg.item()


# ============================
# 4) Eğitim Döngüsü
# ============================

def train_drocc(
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 10,
    R: Optional[float] = None,  # None ise otomatik hesapla
    eps: float = 0.25,
    n_steps: int = 5,
    step_size: Optional[float] = None, # None ise R'ye göre ayarla
    lambda_adv: float = 1.0,
    lr: float = 1e-3,
):
    model = LeNet5DROCC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- OTOMATİK R VE STEP_SIZE AYARI ---
    if R is None:
        # Modeli başlatıp bir radius tahmini yapalım
        # Not: İlk başta rastgele ağırlıklarla radius çok farklı olabilir,
        # bu yüzden DROCC genelde pretrained bir model üzerine kurulur 
        # veya ilk epoch'ta R dinamik güncellenir. 
        # Burada basitlik adına başlangıçta bir kez ölçüyoruz.
        R = estimate_radius(model, train_loader, device=device)
    
    # R çok küçükse (0'a yakınsa) patlamayı önlemek için alt sınır koyalım
    if R < 0.1: 
        R = 0.1
        print("[!] R değeri çok küçük, 0.1'e sabitlendi.")

    if step_size is None:
        # Makale genelde step_size'ı radius'un küçük bir yüzdesi seçer
        step_size = R * 0.05  # Örnek: Yarıçapın %5'i kadar adım at
        print(f"[*] Step size otomatik ayarlandı: {step_size:.4f}")
    # -------------------------------------

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        num_batches = 0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            
            # drocc_loss fonksiyonu artık hesaplanan R ve step_size ile çağrılıyor
            loss, lp, ln = drocc_loss(
                model, x,
                R=R, eps=eps,
                n_steps=n_steps,
                step_size=step_size,
                lambda_adv=lambda_adv,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pos += lp
            total_neg += ln
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_pos = total_pos / num_batches
        avg_neg = total_neg / num_batches
        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | pos={avg_pos:.4f} | neg={avg_neg:.4f} | R={R:.2f}")

    return model


# ============================
# 5) Basit Test / Skor Hesabı
# ============================

@torch.no_grad()
def evaluate_scores(model: LeNet5DROCC, data_loader: DataLoader, device: str = "cuda"):
    model.eval()
    scores = []
    labels = []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        _, logit = model(x)
        prob = torch.sigmoid(logit)
        scores.append(prob.cpu().numpy())
        labels.append(y.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    return scores, labels


# ============================
# 6) main
# ============================

if __name__ == "__main__":
    # -------------------------
    # KLASÖR YAPISI ÖRNEĞİ:
    # data/
    #   train/
    #     normal/   -> sadece normal paketlerden üretilmiş PNG'ler
    #   test/
    #     normal/   -> test için normal
    #     attack/   -> test için saldırı
    # -------------------------

    train_normal_dir = "pcap_images/Normal"
    test_normal_dir = "pcap_images/Normal"  # Test için ayrı klasör oluşturulabilir
    test_attack_dir = "pcap_images/Attack"  # Attack klasörü oluşturulmalı

    batch_size = 64
    img_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Eğitim datası: sadece normal
    train_dataset = PacketImageDataset(train_normal_dir, label=1, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2) Modeli DROCC ile eğit
    model = train_drocc(
    train_loader,
        device=device,
        epochs=10,
        R=None,          # <--- Otomatik hesaplanacak
        eps=0.25,
        n_steps=5,
        step_size=None,  # <--- R'ye göre otomatik ayarlanacak
        lambda_adv=1.0,
        lr=1e-3,
    )

    # 3) İstersen basit test: normal + attack skorları topla
    if os.path.isdir(test_normal_dir) and os.path.isdir(test_attack_dir):

        # İstersen buraya ROC-AUC için sklearn ekleyebilirsin:
        # from sklearn.metrics import roc_auc_score
        # auc = roc_auc_score(labels, scores)
        # print("ROC-AUC:", auc)

        # ... (Yukarıdaki eğitim kodları aynen kalacak) ...

        # 3) DETAYLI TEST VE METRİKLER
        print("\n" + "="*40)
        print("TEST AŞAMASI")
        print("="*40)

        if os.path.isdir(test_normal_dir) and os.path.isdir(test_attack_dir):
            # Test veri setlerini hazırla
            test_normal = PacketImageDataset(test_normal_dir, label=1, img_size=img_size) # Normal=1
            test_attack = PacketImageDataset(test_attack_dir, label=0, img_size=img_size) # Attack=0
            
            test_dataset = ConcatDataset([test_normal, test_attack])
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            print(f"Test Seti Boyutu: {len(test_dataset)} (Normal: {len(test_normal)}, Attack: {len(test_attack)})")

            # Skorları al
            scores, labels = evaluate_scores(model, test_loader, device=device)

            # ---------------------------------------------------------
            # A) ROC-AUC Hesaplama
            # ---------------------------------------------------------
            # Labels: 1 (Normal), 0 (Attack)
            # Scores: Modelin 'normallik' skoru (yüksekse normal, düşükse anomali)
            roc_auc = roc_auc_score(labels, scores)
            print(f"\n[RESULT] ROC-AUC Score: {roc_auc:.4f}")

            # ---------------------------------------------------------
            # B) En İyi Eşik Değerini (Best Threshold) Bulma
            # ---------------------------------------------------------
            fpr, tpr, thresholds = roc_curve(labels, scores)
            # Youden's J istatistiği: J = TPR - FPR
            # En yüksek J değerini veren threshold en idealidir.
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]
            print(f"[INFO] Best Threshold (Youden's J): {best_thresh:.4f}")

            # ---------------------------------------------------------
            # C) Confusion Matrix ve Sınıflandırma Raporu
            # ---------------------------------------------------------
            # Skoru threshold'a göre 0 veya 1'e çevir
            preds = (scores >= best_thresh).astype(int)

            print("\n--- Sınıflandırma Raporu ---")
            # target_names: 0 -> Attack, 1 -> Normal
            print(classification_report(labels, preds, target_names=['Attack', 'Normal']))

            cm = confusion_matrix(labels, preds)
            print("--- Confusion Matrix ---")
            print(cm)
            
            # Confusion Matrix Görselleştirmesi
            plt.figure(figsize=(6, 5))
            # Eğer seaborn yüklü değilse 'pip install seaborn' yapın veya
            # sadece plt.imshow(cm) kullanın. Aşağıdaki seaborn örneğidir:
            try:
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Attack (0)', 'Normal (1)'], 
                            yticklabels=['Attack (0)', 'Normal (1)'])
                plt.ylabel('Gerçek Etiket')
                plt.xlabel('Tahmin Edilen Etiket')
                plt.title('Confusion Matrix')
                plt.savefig("confusion_matrix.png")
                print("[GRAPH] Confusion Matrix kaydedildi: confusion_matrix.png")
            except ImportError:
                print("[!] Seaborn yüklü değil, CM görseli çizilmedi.")
            
            # ROC çizimi (Mevcut kodunuzdaki plt kodu buraya gelecek...)
            print("(Sol Üst: TN (Attack), Sağ Üst: FP, Sol Alt: FN, Sağ Alt: TP (Normal))")

            # ---------------------------------------------------------
            # D) ROC Eğrisi Çizdirme (Tez Görseli)
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (Attack as Normal)')
            plt.ylabel('True Positive Rate (Normal Detection)')
            plt.title('ROC Curve for DROCC Anomaly Detection')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Grafiği kaydet
            save_path = "roc_curve_result.png"
            plt.savefig(save_path)
            print(f"\n[GRAPH] ROC Eğrisi kaydedildi: {save_path}")
            plt.show()

        else:
            print("[!] Test klasörleri bulunamadı, test atlanıyor.")


