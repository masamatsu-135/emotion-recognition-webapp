# train.py
"""
FER-2013 + ResNet-18 / CNN 用の学習スクリプト

前提:
- dataset.py に FER2013ResNetDataset / get_resnet_transforms が定義されている
- model.py に EmotionResNet18 / EmotionCNN / get_model が定義されている
- fer2013.csv が data/raw/fer2013.csv にある

使い方（例）:
    # 軽いCNNで学習（ローカルCPU向き）
    python -m src.training.train --csv_path data/raw/fer2013.csv --model_type cnn

    # ResNet-18で学習（ColabのGPUなど）
    python -m src.training.train --csv_path data/raw/fer2013.csv --model_type resnet --pretrained
"""

import argparse
from pathlib import Path
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FER2013ResNetDataset, get_resnet_transforms
from .model import get_model  # EmotionResNet18 / EmotionCNN を返す想定


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """logits (N, C) と targets (N,) から accuracy を計算"""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    1エポック分の学習
    return: (平均loss, 平均accuracy)
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    # AMP: GPU時のみ有効。CPU時は自動で無効（オーバーヘッド無し）
    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        #imgs = imgs.to(device)
        #labels = labels.to(device)

        #optimizer.zero_grad()

        #outputs = model(imgs)           # (N, num_classes)
        #loss = criterion(outputs, labels)

        #loss.backward()
        #optimizer.step()

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)


        # --- forward (autocast: 混合精度で高速化&省メモリ) ---
        #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):  
            outputs = model(imgs)                  # (N, num_classes)
            loss    = criterion(outputs, labels)

        # --- backward (GradScalerでlossをスケール) ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs, labels) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    評価用（検証 / テスト）
    return: (平均loss, 平均accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val", leave=False):
            #imgs = imgs.to(device)
            #labels = labels.to(device)

            #outputs = model(imgs)
            #loss = criterion(outputs, labels)


            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 評価は backward がないため GradScaler は不要。autocast のみでOK
            #with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)


            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(outputs, labels) * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    output_dir: Path,
    model_type: str,
    filename: str | None = None,
):
    """学習済みモデルを保存"""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = model_type.lower()
    if filename is None:
        filename = f"best_{model_type}_fer2013.pth"

    path = output_dir / filename
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved best model to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/raw/fer2013.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")

    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "resnet"],
        help="使用するモデルタイプ（cnn or resnet）",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="ResNet使用時に ImageNet事前学習重みを使うか（cnnのときは無視）",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model type: {args.model_type}")


    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # 入力サイズが一定なら高速化
        try:
            torch.set_float32_matmul_precision("medium")  # PyTorch 2.x: matmul を高速化
        except Exception:
            pass


    # ===== Dataset / DataLoader =====
    is_cuda = (device.type == "cuda")

    train_dataset = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="Training",
        transform=get_resnet_transforms(train=True),
    )
    val_dataset = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="PublicTest",  # 検証用として PublicTest を使用
        transform=get_resnet_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=(4 if is_cuda else 0),
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(4 if is_cuda else 0),
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val   samples: {len(val_dataset)}")

    # ===== Model / Loss / Optimizer =====
    # CNN のときは pretrained は無視される
    use_pretrained = args.pretrained if args.model_type == "resnet" else False
    model = get_model(
        model_type=args.model_type,
        num_classes=7,
        pretrained=use_pretrained,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ===== 学習ループ =====
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"[INFO] Start training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        print(f"[DEBUG] Start epoch {epoch}")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"({elapsed:.1f} sec)"
        )

        # ベストモデル更新
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, output_dir, model_type=args.model_type)

    print(f"[INFO] Training finished. Best Val Acc: {best_val_acc:.4f}")

    # 一応ベスト重みに戻しておく（後で続けて使う場合に便利）
    model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    main()
