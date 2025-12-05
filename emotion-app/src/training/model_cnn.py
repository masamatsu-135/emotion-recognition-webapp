# model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    FER-2013 (7クラス) 向けのシンプルなCNNモデル
    入力: (N, 1, 48, 48) のグレースケール画像を想定
    出力: (N, num_classes) のロジット
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # 畳み込みブロック1: 1ch -> 32ch
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # 畳み込みブロック2: 32ch -> 64ch
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # 畳み込みブロック3: 64ch -> 128ch
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 48x48 を3回プーリングすると 6x6 になる想定
        # 48 -> 24 -> 12 -> 6
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 48, 48)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)   # -> (N, 32, 24, 24)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)   # -> (N, 64, 12, 12)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)   # -> (N, 128, 6, 6)

        x = x.view(x.size(0), -1)  # フラット化: (N, 128*6*6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # (N, num_classes)

        return x


def get_model(num_classes: int = 7,
              device: str | torch.device = "cpu") -> EmotionCNN:
    """
    学習や推論で使うモデル生成用のヘルパー関数。
    """
    model = EmotionCNN(num_classes=num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # 簡単な動作確認 (python model_cnn.py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device=device)
    model.eval()

    dummy = torch.randn(2, 1, 48, 48).to(device)  # ダミー入力
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)  # -> torch.Size([2, 7]) を想定
