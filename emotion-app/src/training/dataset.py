# dataset.py
"""
FER-2013 用 Dataset （ResNet-18 前提）

- FER-2013 の fer2013.csv を読み込む
- Usage 別に "Training" / "PublicTest" / "PrivateTest" を選択可能
- ResNet-18 用に
    - グレースケール → 3ch
    - 224x224 にリサイズ
    - ImageNet の平均・分散で正規化
を行う transforms を同じファイル内に定義
"""

import csv
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FER2013ResNetDataset(Dataset):
    """
    FER-2013 の CSV を読み込む PyTorch Dataset （ResNet-18 用）

    Args:
        csv_path: fer2013.csv のパス
        usage: "Training", "PublicTest", "PrivateTest" のいずれか
        transform: torchvision.transforms などの前処理
    """
    def __init__(
        self,
        csv_path: str | Path,
        usage: str = "Training",
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.usage = usage
        self.transform = transform

        # (pixels: np.ndarray(48,48), label: int) のリスト
        self.samples: List[Tuple[np.ndarray, int]] = []

        self._load_csv()

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with self.csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Usage でフィルタ
                if row["Usage"] != self.usage:
                    continue

                # emotion: 0〜6 の整数ラベル
                label = int(row["emotion"])

                # pixels: "70 80 82 ..." のスペース区切り文字列
                pixels_str = row["pixels"]
                pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")

                # 48x48 の画像に reshape
                pixels = pixels.reshape(48, 48)

                self.samples.append((pixels, label))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for Usage='{self.usage}'. "
                "Check csv_path or usage string."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pixels, label = self.samples[idx]

        # numpy 配列 (48,48) -> PIL.Image（mode="L": グレースケール）
        img = Image.fromarray(pixels)  # 1ch グレースケール

        # transforms があれば適用（ここで 3ch & 224x224 & 正規化 など）
        if self.transform is not None:
            img = self.transform(img)

        # img: torch.Tensor (3, 224, 224) を想定
        # label: int（0〜6）
        return img, label


def get_resnet_transforms(train: bool = True) -> T.Compose:
    """
    ResNet-18 用の前処理を返す

    - グレースケール -> 3ch 変換
    - 224x224 リサイズ
    - テンソル変換
    - ImageNet の平均・分散で正規化
    - train=True のときは簡単なデータ拡張を追加
    """
    base_transforms = [
        T.Grayscale(num_output_channels=3),  # (1ch) -> (3ch)
        T.Resize((224, 224)),
    ]

    if train:
        # 軽めのデータ拡張（必要に応じて調整してください）
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomRotation(10),  # 回転も入れたければコメントアウトを外す
        ]
        base_transforms = aug + base_transforms

    base_transforms += [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]

    return T.Compose(base_transforms)


if __name__ == "__main__":
    # dataset.py のある場所から見てプロジェクトルートを推定
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]   # emotion-app/ を指す想定

    csv_path = project_root / "data/raw/fer2013.csv"

    train_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="Training",
        transform=get_resnet_transforms(train=True),
    )

    # PublicTest (= validation) 用 Dataset
    val_ds = FER2013ResNetDataset(
        csv_path=csv_path,
        usage="PublicTest",
        transform=get_resnet_transforms(train=False),
    )

    print("Train size :", len(train_ds))
    print("Val size   :", len(val_ds))

    # 1サンプル取り出して形状確認
    img, label = train_ds[0]
    print("Image shape:", img.shape)   # -> torch.Size([3, 224, 224]) のはず
    print("Label      :", label)
