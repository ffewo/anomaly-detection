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
    R: float = 1.0,
    eps: float = 0.25,
    n_steps: int = 5,
    step_size: float = 0.03,
    lambda_adv: float = 1.0,
    lr: float = 1e-3,
):
    model = LeNet5DROCC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        num_batches = 0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
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
        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | pos={avg_pos:.4f} | neg={avg_neg:.4f}")

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
        R=1.0,
        eps=0.25,
        n_steps=5,
        step_size=0.03,
        lambda_adv=1.0,
        lr=1e-3,
    )

    # 3) İstersen basit test: normal + attack skorları topla
    if os.path.isdir(test_normal_dir) and os.path.isdir(test_attack_dir):
        test_normal = PacketImageDataset(test_normal_dir, label=1, img_size=img_size)
        test_attack = PacketImageDataset(test_attack_dir, label=0, img_size=img_size)
        test_dataset = ConcatDataset([test_normal, test_attack])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        scores, labels = evaluate_scores(model, test_loader, device=device)

        # Örnek: Eşik belirlemeden sadece istatistikleri yazdır
        print("Normal skor ort:", scores[labels == 1].mean())
        print("Attack skor ort:", scores[labels == 0].mean())

        # İstersen buraya ROC-AUC için sklearn ekleyebilirsin:
        # from sklearn.metrics import roc_auc_score
        # auc = roc_auc_score(labels, scores)
        # print("ROC-AUC:", auc)
