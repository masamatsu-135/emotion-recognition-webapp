import torch
import torch.nn as nn
from torchvision import models


class EmotionResNet18(nn.Module):
    """
    FER-2013 (7クラス) 向けの ResNet-18 ベース感情認識モデル
    入力: (N, 3, 224, 224) のRGB画像を想定
    出力: (N, num_classes) のロジット
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()

        if pretrained:
            # PyTorch 2.x 以降の推奨書き方
            try:
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except AttributeError:
                # 古いバージョンの場合のフォールバック
                backbone = models.resnet18(pretrained=True)
        else:
            try:
                backbone = models.resnet18(weights=None)
            except TypeError:
                backbone = models.resnet18(pretrained=False)

        # 最後の全結合層（fc）を付け替えて、7クラス分類用にする
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, 224, 224)
        return: (N, num_classes)
        """
        return self.backbone(x)


class EmotionCNN(nn.Module):
    """
    FER-2013 (7クラス) 向けの軽量CNNモデル
    入力: (N, 3, 224, 224) のRGB画像を想定
    出力: (N, num_classes) のロジット

    ※ dataset.py 側の transforms で
       - Grayscale(num_output_channels=3)
       - Resize((224, 224))
       を行っている前提
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # 畳み込みブロック
        self.features = nn.Sequential(
            # block1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            # block2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            # block3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
        )

        # 空間平均で 1x1 に圧縮（入力サイズに依存しない形に）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (N, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                # -> (N, 128)
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def get_model(
    model_type: str = "cnn",
    num_classes: int = 7,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    学習や推論で使うモデル生成用のヘルパー関数。
    model_type に応じて CNN / ResNet を切り替える。

    model_type:
        - "cnn"   : EmotionCNN（軽量・CPU向き）を返す
        - "resnet": EmotionResNet18（高精度）を返す
    """
    model_type = model_type.lower()
    if model_type == "resnet":
        model = EmotionResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "cnn":
        # CNN には事前学習はないので pretrained は無視
        model = EmotionCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'cnn' or 'resnet'.")

    model = model.to(device)
    return model


if __name__ == "__main__":
    # 簡単な動作確認用 (python model.py で実行)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Test CNN model ===")
    cnn = get_model(model_type="cnn", device=device)
    cnn.eval()
    dummy_cnn = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out_cnn = cnn(dummy_cnn)
    print("CNN Output shape:", out_cnn.shape)  # -> torch.Size([2, 7])

    print("=== Test ResNet18 model ===")
    resnet = get_model(model_type="resnet", device=device, pretrained=False)
    resnet.eval()
    dummy_resnet = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out_resnet = resnet(dummy_resnet)
    print("ResNet Output shape:", out_resnet.shape)  # -> torch.Size([2, 7])
