# predictor.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# training パッケージからモデル＆前処理を import
from ..training.model import get_model
from ..training.dataset import get_resnet_transforms

# 同じ inference パッケージ内の labels.py からラベルを import
from .labels import EMOTION_LABELS_EN, EMOTION_LABELS_JA, get_label_en, get_label_ja


ModelType = Literal["cnn", "resnet"]


class EmotionPredictor:
    """
    学習済みモデル (.pth) を読み込んで、
    顔画像1枚から感情を推論するクラス。

    想定する入力:
        - 顔だけが写った画像 (RGB) の PIL.Image.Image
        - または np.ndarray (H, W, 3) の RGB / BGR 画像

    使い方:
        predictor = EmotionPredictor(
            model_type="cnn",
            checkpoint_path="models/checkpoints/best_cnn_fer2013.pth",
        )

        result = predictor.predict_from_pil(face_image_pil)
        # result["label_ja"] -> "幸福" など
    """

    def __init__(
        self,
        model_type: ModelType = "cnn",
        checkpoint_path: str | Path = "models/checkpoints/best_cnn_fer2013.pth",
        device: Optional[str] = None,
    ) -> None:
        self.model_type: ModelType = model_type

        # デバイス決定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 推論時の transforms（学習時の val 用と同じもの）
        # augmentation は不要なので train=False
        self.transform = get_resnet_transforms(train=False)

        # モデルの初期化（学習済み重みを上書きロードするので pretrained=False でOK）
        self.model = get_model(
            model_type=model_type,
            num_classes=7,
            pretrained=False,
            device=self.device,
        )

        # 学習済み重みのロード
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "train.py で学習を行い、.pth ファイルを作成してください。"
            )

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[INFO] Loaded model_type='{model_type}' from: {checkpoint_path}")
        print(f"[INFO] Using device: {self.device}")

    # ==============================
    # 画像前処理まわり
    # ==============================

    def _to_pil(self, image: Image.Image | np.ndarray, bgr: bool = False) -> Image.Image:
        """
        入力を PIL.Image に統一するヘルパー関数。

        Args:
            image: PIL.Image.Image または np.ndarray (H, W, 3)
            bgr:   OpenCV(BGR) 形式の配列なら True
        """
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, np.ndarray):
            # H, W, C を想定
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected image shape (H, W, 3), got {image.shape}"
                )

            img = image
            if bgr:
                # OpenCV など BGR → RGB に変換
                img = img[:, :, ::-1]

            return Image.fromarray(img.astype(np.uint8))

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Use PIL.Image.Image or np.ndarray."
        )

    def _preprocess(self, image: Image.Image | np.ndarray, bgr: bool = False) -> torch.Tensor:
        """
        1枚の画像をモデル入力用の Tensor (1, 3, 224, 224) に変換。
        """
        pil_img = self._to_pil(image, bgr=bgr)
        tensor = self.transform(pil_img)  # (3, 224, 224)
        tensor = tensor.unsqueeze(0)      # (1, 3, 224, 224)
        return tensor.to(self.device)

    # ==============================
    # 推論本体
    # ==============================

    def predict_from_pil(self, image: Image.Image) -> Dict[str, Any]:
        """
        顔だけが写った PIL 画像から感情を推論する。

        Returns:
            {
                "class_id": int,
                "label_en": str,
                "label_ja": str,
                "probs": List[float],   # 各クラスの確率 (長さ7)
            }
        """
        input_tensor = self._preprocess(image, bgr=False)
        return self._predict_from_tensor(input_tensor)

    def predict_from_ndarray(self, image: np.ndarray, bgr: bool = False) -> Dict[str, Any]:
        """
        顔だけが写った numpy 画像から感情を推論する。

        Args:
            image: np.ndarray (H, W, 3)
            bgr:   image が BGR(OpenCV) 形式なら True, RGBなら False

        Returns:
            predict_from_pil と同じ dict
        """
        input_tensor = self._preprocess(image, bgr=bgr)
        return self._predict_from_tensor(input_tensor)

    def _predict_from_tensor(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        前処理済み Tensor (1, 3, 224, 224) から推論を行う内部メソッド。
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)         # (1, 7)
            probs = F.softmax(logits, dim=1)[0]       # (7,)

        class_id = int(torch.argmax(probs).item())
        probs_list = probs.cpu().numpy().tolist()

        result = {
            "class_id": class_id,
            "label_en": get_label_en(class_id),
            "label_ja": get_label_ja(class_id),
            "probs": probs_list,
        }
        return result


if __name__ == "__main__":
    """
    簡単な動作確認用。

    例:
        python -m src.inference.predictor

    実行前に、適当な顔画像を用意して、
    image_path を差し替えてテストしてください。
    （まだ face_detector.py を作っていないので、
      ここでは「すでに顔だけにトリミングされている」画像を想定）
    """
    import os

    # プロジェクトルートを想定したパス例
    DEFAULT_CHECKPOINT = Path("models/checkpoints/best_cnn_fer2013.pth")

    # テスト用のダミー画像パス（実際の画像に差し替えてください）
    # 例: image_path = Path("tests/sample_face.jpg")
    image_path = Path("tests/face.jpg")

    if image_path is None or not Path(image_path).exists():
        print("[WARN] テスト用の顔画像パスを predictor.py 内で設定してください。")
    else:
        predictor = EmotionPredictor(
            model_type="cnn",
            checkpoint_path=DEFAULT_CHECKPOINT,
            device=None,  # 自動で CPU / GPU を選択
        )
        img = Image.open(image_path).convert("RGB")
        result = predictor.predict_from_pil(img)
        print("Prediction result:", result)
