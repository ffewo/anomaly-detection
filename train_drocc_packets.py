import os
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)

# tqdm opsiyonel
try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    class _NoTqdm:
        def __init__(self, iterable, desc: str = ""):
            self.iterable = iterable
            self.desc = desc
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            pass
    def tqdm(iterable, desc: str = ""):
        return _NoTqdm(iterable, desc=desc)


# ============================
# 1) Dataset
# ============================

class PacketImageDataset(Dataset):
    """
    32x32 Grayscale PNG/JPG görüntülerini okur ve Tensor'e çevirir.
    label: bu klasörden gelen tüm örneklerin etiketi
    """
    def __init__(self, root_dir: str, label: int, img_size: int = 32):
        self.root_dir = root_dir
        self.label = int(label)
        self.img_size = img_size

        if not os.path.exists(root_dir):
            raise RuntimeError(f"Klasör bulunamadı: {root_dir}")

        exts = (".png", ".jpg", ".jpeg")
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(exts)
        ]
        self.files.sort()

        if len(self.files) == 0:
            print(f"[UYARI] '{root_dir}' içinde görüntü yok!")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
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
            print(f"Hata: {img_path} okunamadı -> {e}")
            return torch.zeros((1, self.img_size, self.img_size)), self.label


# ============================
# 2) Model
# ============================

class LeNet5DROCC(nn.Module):
    """
    LeNet-5 Backbone + tek çıkış (logit).
    Çıkış logit: sigmoid(logit) ~ normal olasılığı gibi yorumlanabilir.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2)

        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        x = torch.tanh(self.conv3(x))

        x = x.view(x.size(0), -1)
        emb = torch.tanh(self.fc1(x))
        logit = self.fc2(emb)
        return logit.squeeze(-1)


# ============================
# 3) DROCC Input-space helpers
# ============================

def project_input_to_sphere(x, x_adv, R, eps=0.25):
    diff = x_adv - x
    diff_flat = diff.view(diff.size(0), -1)
    dist = diff_flat.norm(p=2, dim=1, keepdim=True) + 1e-8

    lower = R
    upper = (1.0 + eps) * R

    scale = torch.clamp(dist, min=lower, max=upper) / dist
    scale = scale.view(dist.size(0), 1, 1, 1)

    x_proj = x + diff * scale
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
    device = x.device
    B = x.size(0)

    # 1) Pozitif (Normal) loss: hedef 1
    logit_pos = model(x)
    y_pos = torch.ones(B, device=device)
    loss_pos = F.binary_cross_entropy_with_logits(logit_pos, y_pos)

    # 2) Adversarial negatif üretimi (input space)
    model.eval()

    x_adv = x.detach().clone()
    x_adv = x_adv + torch.randn_like(x_adv) * 0.01
    x_adv = project_input_to_sphere(x, x_adv, R, eps)
    x_adv.requires_grad_(True)

    for _ in range(n_steps):
        logit_adv = model(x_adv)
        # “Normal” demesini maksimize ederek zor örnek arıyoruz
        loss_search = F.binary_cross_entropy_with_logits(logit_adv, y_pos)
        grad = torch.autograd.grad(loss_search, x_adv)[0]
        x_adv = x_adv + step_size * torch.sign(grad)
        x_adv = project_input_to_sphere(x, x_adv, R, eps)
        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)

    model.train()

    # 3) Negatif loss: hedef 0
    x_adv = x_adv.detach()
    logit_neg = model(x_adv)
    y_neg = torch.zeros(B, device=device)
    loss_neg = F.binary_cross_entropy_with_logits(logit_neg, y_neg)

    total_loss = loss_pos + lambda_adv * loss_neg
    return total_loss, float(loss_pos.item()), float(loss_neg.item())


def estimate_radius_input_space(data_loader, device="cuda"):
    print("[*] Radius (R) tahmini yapılıyor...")
    distances = []

    for i, (x, _) in enumerate(data_loader):
        if i > 5:
            break
        x = x.to(device)
        B = x.size(0)
        if B < 2:
            continue

        x_flat = x.view(B, -1)
        center = x_flat.mean(dim=0, keepdim=True)
        dist = torch.norm(x_flat - center, p=2, dim=1)
        distances.append(dist)

    if len(distances) == 0:
        # çok küçük dataset olursa fallback
        print("[UYARI] R tahmini için yeterli batch yok. R=1.0 fallback.")
        return 1.0

    all_dists = torch.cat(distances)
    estimated_R = all_dists.mean().item() * 0.20
    print(f"[*] Tahmin edilen R: {estimated_R:.6f}")
    return float(estimated_R)


# ============================
# 4) Train / Evaluate
# ============================

def train_model(model, train_loader, device, epochs=10, R=None, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if R is None:
        R = estimate_radius_input_space(train_loader, device)

    step_size = max(R * 0.1, 1e-6)

    print(f"[*] Eğitim Başlıyor... (Epochs={epochs}, R={R:.6f}, step={step_size:.6f})")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()

            loss, lp, ln = drocc_loss_input_space(
                model, x, R=R, n_steps=5, step_size=step_size
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Lpos": f"{lp:.4f}", "Lneg": f"{ln:.4f}"})

        print(f"Epoch {epoch} -> Avg Loss: {total_loss / max(len(train_loader),1):.6f}")

    return model


@torch.no_grad()
def evaluate_scores(model, test_loader, device):
    """
    scores_normal_prob: sigmoid(logit)  (yüksek = daha normal)
    scores_anomaly:     1 - sigmoid(logit) (yüksek = daha saldırı/anomali)
    """
    model.eval()
    scores_normal_prob = []
    labels = []

    print("[*] Test ediliyor...")
    for x, y in tqdm(test_loader, desc="Test"):
        x = x.to(device)
        logit = model(x)
        prob_normal = torch.sigmoid(logit)  # [0,1]
        scores_normal_prob.extend(prob_normal.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())

    scores_normal_prob = np.array(scores_normal_prob, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)

    scores_anomaly = 1.0 - scores_normal_prob
    return scores_normal_prob, scores_anomaly, labels


def pick_threshold_youden(labels_attack, scores_anomaly):
    """
    ROC curve thresholds içinde `inf` olabiliyor.
    Bu fonksiyon inf'i eleleyip Youden J ile en iyi eşiği seçer.
    """
    fpr, tpr, thresholds = roc_curve(labels_attack, scores_anomaly)

    finite = np.isfinite(thresholds)
    if finite.sum() == 0:
        # aşırı uç durum
        return np.max(scores_anomaly)

    fpr_f = fpr[finite]
    tpr_f = tpr[finite]
    thr_f = thresholds[finite]

    J = tpr_f - fpr_f
    best_idx = int(np.argmax(J))
    return float(thr_f[best_idx])


# ============================
# 5) Main
# ============================

if __name__ == "__main__":

    # --- KLASÖRLER ---
    TRAIN_NORMAL_DIR = "pcap_images/Normal"
    TEST_NORMAL_DIR  = "pcap_images/test"
    TEST_ATTACK_DIR  = "pcap_images/Attacks"

    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Cihaz: {DEVICE}")

    if not os.path.exists(TRAIN_NORMAL_DIR):
        print(f"HATA: {TRAIN_NORMAL_DIR} bulunamadı! Lütfen yolu düzeltin.")
        raise SystemExit(1)

    # Train: sadece normal (label=1)
    train_ds = PacketImageDataset(TRAIN_NORMAL_DIR, label=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LeNet5DROCC().to(DEVICE)
    model = train_model(model, train_loader, device=DEVICE, epochs=EPOCHS, lr=LR, R=None)

    # Test
    if os.path.exists(TEST_NORMAL_DIR) and os.path.exists(TEST_ATTACK_DIR):
        test_norm_ds = PacketImageDataset(TEST_NORMAL_DIR, label=1)  # Normal=1
        test_att_ds  = PacketImageDataset(TEST_ATTACK_DIR, label=0)  # Attack=0

        test_ds = ConcatDataset([test_norm_ds, test_att_ds])
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"\nTest seti: {len(test_norm_ds)} Normal, {len(test_att_ds)} Saldırı.")

        scores_normal_prob, scores_anomaly, labels = evaluate_scores(model, test_loader, DEVICE)

        # Attack pozitif yapalım (Attack=1, Normal=0)
        labels_attack = 1 - labels  # çünkü labels: Normal=1, Attack=0

        # Debug: skor yönü doğru mu?
        if (labels_attack == 0).any() and (labels_attack == 1).any():
            print("\n[DEBUG]")
            print("Anomaly score min/max:", scores_anomaly.min(), scores_anomaly.max())
            print("Normal mean anomaly:", scores_anomaly[labels_attack == 0].mean())
            print("Attack  mean anomaly:", scores_anomaly[labels_attack == 1].mean())
            print("Normal mean prob  :", scores_normal_prob[labels_attack == 0].mean())
            print("Attack  mean prob  :", scores_normal_prob[labels_attack == 1].mean())

        # AUC (Attack pozitif) - DOĞRUDAN anomaly score ile
        auc_attack = roc_auc_score(labels_attack, scores_anomaly)

        # AUC (Normal pozitif) - normal prob ile
        auc_normal = roc_auc_score(labels, scores_normal_prob)

        print(f"\n[SONUÇ] ROC-AUC (Attack=pozitif, anomaly_score): {auc_attack:.4f}")
        print(f"[SONUÇ] ROC-AUC (Normal=pozitif, normal_prob):  {auc_normal:.4f}")

        # Threshold seç (Youden J) - inf problemi çözüldü
        best_thresh = pick_threshold_youden(labels_attack, scores_anomaly)
        print(f"[BİLGİ] En iyi eşik (anomaly_score >= thresh => Attack): {best_thresh:.6f}")

        # Tahmin: 1=Attack, 0=Normal (attack-pozitif düzeninde)
        preds_attack = (scores_anomaly >= best_thresh).astype(int)

        # Rapor için tekrar senin label düzenine döndürelim:
        # labels: Normal=1, Attack=0
        preds_labels = 1 - preds_attack

        print("\n--- Sınıflandırma Raporu (labels: Attack=0, Normal=1) ---")
        print(classification_report(labels, preds_labels, target_names=["Saldırı (0)", "Normal (1)"]))

        cm = confusion_matrix(labels, preds_labels)

        # ROC curve (attack-pozitif)
        fpr, tpr, _ = roc_curve(labels_attack, scores_anomaly)

        # Plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        try:
            import seaborn as sns
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Saldırı", "Normal"],
                yticklabels=["Saldırı", "Normal"]
            )
        except ImportError:
            plt.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.xticks([0, 1], ["Saldırı", "Normal"])
            plt.yticks([0, 1], ["Saldırı", "Normal"])
        plt.title("Confusion Matrix")
        plt.xlabel("Tahmin")
        plt.ylabel("Gerçek")

        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, lw=2, label=f"AUC (Attack+) = {auc_attack:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Eğrisi (Attack pozitif, anomaly_score)")
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig("drocc_sonuclar.png", dpi=200)
        print("[KAYIT] Grafikler 'drocc_sonuclar.png' dosyasına kaydedildi.")
        plt.show()

    else:
        print("Test klasörleri eksik, test yapılmadı.")
