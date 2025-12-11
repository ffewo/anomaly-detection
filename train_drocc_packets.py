import os
import argparse
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

# İlerleme çubuğu (Opsiyonel ama Input Space yavaş olduğu için önerilir)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""): return iterable

# ============================
# 1) Dataset & DataLoader
# ============================

class PacketImageDataset(Dataset):
    """
    32x32 Grayscale PNG görüntülerini okur ve Tensör'e çevirir.
    """
    def __init__(self, root_dir: str, label: int, img_size: int = 32):
        self.root_dir = root_dir
        self.label = label
        self.img_size = img_size
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Klasör bulunamadı: {root_dir}")

        exts = (".png", ".jpg", ".jpeg")
        self.files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) 
            if f.lower().endswith(exts)
        ]
        self.files.sort()

        if len(self.files) == 0:
            print(f"[UYARI] '{root_dir}' içinde görüntü yok!")

        # 32x32, Grayscale, 0-1 arası Tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.files[idx]
        try:
            img = Image.open(img_path)
            img = self.transform(img)
            return img, self.label
        except Exception as e:
            # Bozuk dosya varsa siyah resim dön (Kodu kırmamak için)
            print(f"Hata: {img_path} okunamadı -> {e}")
            return torch.zeros((1, self.img_size, self.img_size)), self.label


# ============================
# 2) LeNet-5 Modeli
# ============================

class LeNet5DROCC(nn.Module):
    """
    LeNet-5 Backbone + Tek Çıkışlı (One-Class) Head
    Giriş: (Batch, 1, 32, 32)
    Çıkış: (Batch, 1) -> Logit Değeri
    """
    def __init__(self):
        super().__init__()
        # Feature Extractor (Backbone)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        # Classifier (Head)
        # 120 -> 84 -> 1 (Logit)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 1) # DROCC tek bir skor üretir

    def forward(self, x):
        # C1 -> Pool
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        # C2 -> Pool
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        # C3
        x = torch.tanh(self.conv3(x))
        
        # Flatten (Batch, 120)
        x = x.view(x.size(0), -1)
        
        # F6 (Embedding)
        emb = torch.tanh(self.fc1(x))
        
        # Output (Logit)
        logit = self.fc2(emb)
        
        return logit.squeeze(-1)


# ============================
# 3) Input Space DROCC Fonksiyonları
# ============================

def project_input_to_sphere(x, x_adv, R, eps=0.25):
    """
    Adversarial örneği (x_adv), orijinal örneğin (x) etrafındaki
    R ve (1+eps)R yarıçaplı halkanın içine çeker (Projection).
    """
    # Piksel farkları
    diff = x_adv - x
    # Norm hesabı için düzleştir (Batch, 1024)
    diff_flat = diff.view(diff.size(0), -1)
    
    # Euclidean Norm (Mesafe)
    dist = diff_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
    
    # Halka sınırları
    lower = R
    upper = (1.0 + eps) * R
    
    # Scale faktörü (Eğer mesafe halkanın dışındaysa içine çeker)
    scale = torch.clamp(dist, min=lower, max=upper) / dist
    
    # Boyut düzeltme (Batch, 1, 1, 1) broadcast için
    scale = scale.view(*dist.shape, 1, 1)
    
    # x_adv güncelle
    x_proj = x + diff * scale
    
    # Görüntü olduğu için [0, 1] aralığına sıkıştır (Clipping)
    x_proj = torch.clamp(x_proj, 0.0, 1.0)
    
    return x_proj


def drocc_loss_input_space(
    model: nn.Module,
    x: torch.Tensor,
    R: float,
    eps: float = 0.25,
    n_steps: int = 5,
    step_size: float = 0.05,
    lambda_adv: float = 1.0
):
    """
    Girdi uzayında (Input Space) adversarial negatif üreterek loss hesaplar.
    """
    device = x.device
    batch_size = x.size(0)
    
    # --- 1) Pozitif (Normal) Loss ---
    logit_pos = model(x)
    # Normallerin etiketi 1'dir
    y_pos = torch.ones(batch_size, device=device)
    loss_pos = F.binary_cross_entropy_with_logits(logit_pos, y_pos)
    
    # --- 2) Negatif (Attack) Üretimi ---
    # Gradient Ascent için modeli eval moduna al (BatchNorm vs. etkilenmesin)
    model.eval()
    
    # x'in kopyasını al ve rastgele gürültü ekle
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.randn_like(x_adv) * 0.01 
    
    # İlk projeksiyon (R mesafesine at)
    x_adv = project_input_to_sphere(x, x_adv, R, eps)
    x_adv.requires_grad_(True)
    
    # Döngü: Modeli "Normal" dedirtmeye zorlayarak en zor örneği bul
    for _ in range(n_steps):
        logit_adv = model(x_adv)
        
        # Hedef: Model buna 1 (Normal) desin ki, biz tersine (anomaliye) itelim
        loss_search = F.binary_cross_entropy_with_logits(logit_adv, y_pos)
        
        # Piksellere göre türev al
        grad = torch.autograd.grad(loss_search, x_adv)[0]
        
        # Gradient Ascent (Zor örneğe doğru git) - FGSM mantığı (sign)
        x_adv = x_adv + step_size * torch.sign(grad)
        
        # Tekrar manifoldun (halkanın) içine çek
        x_adv = project_input_to_sphere(x, x_adv, R, eps)
        
        # Döngü hazırlığı
        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)
        
    # --- 3) Negatif (Attack) Loss ---
    # Eğitime geri dön
    model.train()
    
    x_adv = x_adv.detach() # Artık x_adv sabit
    logit_neg = model(x_adv)
    
    # Negatiflerin etiketi 0'dır
    y_neg = torch.zeros(batch_size, device=device)
    loss_neg = F.binary_cross_entropy_with_logits(logit_neg, y_neg)
    
    # Toplam Loss
    total_loss = loss_pos + lambda_adv * loss_neg
    
    return total_loss, loss_pos.item(), loss_neg.item()


def estimate_radius_input_space(data_loader, device="cuda"):
    """
    Veri setindeki rastgele görüntü çiftleri arasındaki ortalama mesafeyi hesaplar.
    Bu, R (Radius) parametresini başlatmak için iyi bir yöntemdir.
    """
    print("[*] Radius (R) tahmini yapılıyor...")
    distances = []
    
    # Sadece ilk 5 batch yeterli
    for i, (x, _) in enumerate(data_loader):
        if i > 5: break
        x = x.to(device)
        B = x.size(0)
        if B < 2: continue
        
        # Flatten
        x_flat = x.view(B, -1)
        
        # Batch içindeki her elemanın ortalamaya olan uzaklığı
        center = x_flat.mean(dim=0, keepdim=True)
        dist = torch.norm(x_flat - center, p=2, dim=1)
        distances.append(dist)
        
    all_dists = torch.cat(distances)
    # Ortalama mesafenin biraz fazlası güvenli bir R değeridir
    estimated_R = all_dists.mean().item() * 1.5
    print(f"[*] Tahmin edilen R: {estimated_R:.4f}")
    return estimated_R


# ============================
# 4) Eğitim ve Test Döngüleri
# ============================

def train_model(
    model, train_loader, device, 
    epochs=10, R=None, lr=1e-3
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # R verilmediyse tahmin et
    if R is None:
        R = estimate_radius_input_space(train_loader, device)
    
    # Step size genelde R'nin küçük bir yüzdesidir
    step_size = R * 0.1 
    
    print(f"[*] Eğitim Başlıyor... (Epochs: {epochs}, R: {R:.2f}, Step: {step_size:.2f})")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_pos = 0
        total_neg = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            
            # Input Space Loss
            loss, lp, ln = drocc_loss_input_space(
                model, x, R=R, n_steps=5, step_size=step_size
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pos += lp
            total_neg += ln
            
            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "L_Pos": f"{lp:.3f}", "L_Neg": f"{ln:.3f}"})
            
        print(f"Epoch {epoch} Bitti -> Avg Loss: {total_loss/len(train_loader):.4f}")
        
    return model


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    scores = []
    labels = []
    
    print("[*] Test Ediliyor...")
    for x, y in tqdm(test_loader, desc="Test"):
        x = x.to(device)
        logit = model(x)
        # Sigmoid ile 0-1 arasına çek (Olasılık)
        prob = torch.sigmoid(logit)
        
        scores.extend(prob.cpu().numpy())
        labels.extend(y.numpy())
        
    return np.array(scores), np.array(labels)


# ============================
# 5) Main
# ============================

if __name__ == "__main__":
    
    # --- AYARLAR ---
    # PCAP veya CSV kodundan gelen çıktı klasörlerini buraya yazın:
    TRAIN_NORMAL_DIR = "data/train/normal"
    TEST_NORMAL_DIR = "data/test/normal"
    TEST_ATTACK_DIR = "data/test/attack"
    
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Cihaz: {DEVICE}")
    
    # 1. Veri Yükleme
    # Eğer klasörler yoksa hata vermemesi için kontrol edelim
    if not os.path.exists(TRAIN_NORMAL_DIR):
        print(f"HATA: {TRAIN_NORMAL_DIR} bulunamadı! Lütfen yolu düzeltin.")
        exit()

    train_ds = PacketImageDataset(TRAIN_NORMAL_DIR, label=1) # Normal=1
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. Model Kurulumu
    model = LeNet5DROCC().to(DEVICE)
    
    # 3. Eğitim (Input Space)
    model = train_model(
        model, train_loader, device=DEVICE, 
        epochs=EPOCHS, lr=LR, R=None # R'yi otomatik bulur
    )
    
    # 4. Test Hazırlığı
    if os.path.exists(TEST_NORMAL_DIR) and os.path.exists(TEST_ATTACK_DIR):
        test_norm_ds = PacketImageDataset(TEST_NORMAL_DIR, label=1) # Normal=1
        test_att_ds = PacketImageDataset(TEST_ATTACK_DIR, label=0)  # Attack=0
        
        test_ds = ConcatDataset([test_norm_ds, test_att_ds])
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"\nTest Seti: {len(test_norm_ds)} Normal, {len(test_att_ds)} Saldırı.")
        
        # 5. Değerlendirme
        scores, labels = evaluate(model, test_loader, DEVICE)
        
        # ROC-AUC
        auc = roc_auc_score(labels, scores)
        print(f"\n[SONUÇ] ROC-AUC Score: {auc:.4f}")
        
        # Best Threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_thresh = thresholds[best_idx]
        print(f"[BİLGİ] En iyi Eşik Değeri (Threshold): {best_thresh:.4f}")
        
        # Tahminler (0 veya 1)
        preds = (scores >= best_thresh).astype(int)
        
        # Rapor
        print("\n--- Sınıflandırma Raporu ---")
        print(classification_report(labels, preds, target_names=["Saldırı (0)", "Normal (1)"]))
        
        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        
        # Görselleştirme
        plt.figure(figsize=(10, 4))
        
        # 1. Grafik: Confusion Matrix
        plt.subplot(1, 2, 1)
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Saldırı", "Normal"],
                        yticklabels=["Saldırı", "Normal"])
        except ImportError:
            plt.imshow(cm, cmap="Blues")
            plt.text(0,0, str(cm[0,0]), color="red") # Basit fallback
        plt.title("Confusion Matrix")
        plt.xlabel("Tahmin")
        plt.ylabel("Gerçek")
        
        # 2. Grafik: ROC Curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Eğrisi')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig("drocc_sonuclar.png")
        print("[KAYIT] Grafikler 'drocc_sonuclar.png' dosyasına kaydedildi.")
        plt.show()
        
    else:
        print("Test klasörleri eksik, test yapılmadı.")