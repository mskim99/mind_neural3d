import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open_clip
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


class EEGCLIPAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim, clip_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        features = self.net(x)
        return F.normalize(features, dim=-1)


def get_text_embeddings(dataset, clip_model, clip_tokenizer, device):
    categories = []
    for c in range(72):
        name = str(dataset.name_list[c, 0])[3:].replace("_", " ")
        categories.append(f"a photo of a {name}")

    with torch.no_grad():
        tokens = clip_tokenizer(categories).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features, categories


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG-CLIP Aligner")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="./checkpoints", help="Directory to save the model and logs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/data/jionkim/neuro_3D/"
    sub_id = "sub01"

    # 저장 폴더 안전 생성
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # [PATCH] 로그 파일 초기화 (CSV 헤더 작성)
    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Loss,Train_Acc,LR\n")
    print(f"[*] Training log will be saved to: {log_path}")

    print("[*] Loading OpenCLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    train_dataset = AllDataFeatureTwoEEG(
        data_path=data_path, sub_list=[sub_id], train=True,
        num_frames=6, rendered_view_path=f"{data_path}/render_grid_v4"
    )
    if hasattr(AllDataFeatureTwoEEG, "_validate_rendered_dataset"):
        AllDataFeatureTwoEEG._validate_rendered_dataset = lambda self: None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    text_embeddings, class_names = get_text_embeddings(train_dataset, clip_model, clip_tokenizer, device)

    static_train_path = Path(data_path) / "EEGdata" / sub_id / f"{sub_id}_train_data_1s_250Hz.npy"
    static_train = np.load(static_train_path, mmap_mode="r")

    expected_shape = (1, 72, 8, 2, 64, 250)
    if tuple(static_train.shape) == expected_shape[1:]:
        static_train = static_train[None]
    elif tuple(static_train.shape) != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tuple(static_train.shape)}")

    static_train = np.asarray(static_train).mean(axis=3)
    input_dim = static_train.shape[-2] * 2

    model = EEGCLIPAligner(input_dim=input_dim, hidden_dim=1024, clip_dim=512).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc = 0.0
    best_model_path = os.path.join(args.out_dir, "eeg_clip_aligner_best.pt")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()

            eeg_raw = []
            for i in range(len(batch["cls_index"])):
                s, c, o = int(batch["subject_index"][i]), int(batch["cls_index"][i]), int(batch["obj_index"][i])
                eeg_raw.append(static_train[s, c, o])
            eeg_raw = np.stack(eeg_raw)

            eeg_mean = torch.from_numpy(eeg_raw.mean(axis=-1)).float().to(device)
            eeg_std = torch.from_numpy(eeg_raw.std(axis=-1)).float().to(device)
            features = torch.cat([eeg_mean, eeg_std], dim=-1)

            targets = batch["cls_index"].to(device)
            target_text_embeds = text_embeddings[targets]

            eeg_embeds = model(features)

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * eeg_embeds @ target_text_embeds.t()
            labels = torch.arange(len(logits), device=device)

            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f} | LR: {current_lr:.6f}")

        # [PATCH] 매 에폭마다 로그 파일에 기록 추가
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{acc:.6f},{current_lr:.6f}\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] Best model saved to {best_model_path} (Acc: {best_acc:.4f})")

    last_model_path = os.path.join(args.out_dir, "eeg_clip_aligner_last.pt")
    torch.save(model.state_dict(), last_model_path)
    print(f"[*] Training complete. Final model saved to {last_model_path}")


if __name__ == "__main__":
    main()