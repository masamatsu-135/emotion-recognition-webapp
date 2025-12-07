# face_detector.py

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class FaceBox:
    """
    1つの顔領域を表す矩形 (x, y, w, h)
    x, y: 左上座標
    w, h: 幅と高さ
    """
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _get_face_cascade() -> cv2.CascadeClassifier:
    """
    OpenCV 同梱の Haar Cascade を読み込むヘルパー関数。
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return face_cascade


def detect_faces(
    image: np.ndarray,
    bgr: bool = True,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> List[FaceBox]:
    """
    画像から顔を検出し、FaceBox のリストを返す。

    Args:
        image: np.ndarray, shape=(H, W, 3)
        bgr:   image が OpenCV の BGR 形式なら True, RGB なら False
        scale_factor, min_neighbors, min_size:
            Haar Cascade のパラメータ（必要に応じて調整）

    Returns:
        FaceBox のリスト（見つからなければ空リスト）
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image.shape}")

    img = image.copy()

    # BGR / RGB に応じてグレースケール化
    if bgr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = _get_face_cascade()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    boxes: List[FaceBox] = [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    return boxes


def crop_faces(
    image: np.ndarray,
    boxes: List[FaceBox],
) -> List[np.ndarray]:
    """
    元画像とFaceBoxリストから、顔領域だけを切り出した画像を返す。

    Args:
        image: 元画像 (H, W, 3)
        boxes: FaceBox のリスト

    Returns:
        顔だけの画像 np.ndarray (h, w, 3) のリスト
        （image と同じ色空間のまま返す：BGRならBGR、RGBならRGB）
    """
    crops: List[np.ndarray] = []
    h, w, _ = image.shape

    for box in boxes:
        x1 = max(box.x, 0)
        y1 = max(box.y, 0)
        x2 = min(box.x + box.w, w)
        y2 = min(box.y + box.h, h)

        if x2 > x1 and y2 > y1:
            face_img = image[y1:y2, x1:x2, :]
            crops.append(face_img)

    return crops


def detect_and_crop_largest_face(
    image: np.ndarray,
    bgr: bool = True,
    **detect_kwargs,
) -> Tuple[Optional[FaceBox], Optional[np.ndarray]]:
    """
    画像から「最も大きい顔」を1つだけ検出して、そのbboxと切り出し画像を返す。

    Args:
        image: np.ndarray (H, W, 3)
        bgr:   image が BGR 形式かどうか
        detect_kwargs: detect_faces に渡す追加パラメータ

    Returns:
        (FaceBox or None, face_image or None)
        顔が見つからなければ (None, None)
    """
    boxes = detect_faces(image, bgr=bgr, **detect_kwargs)
    if not boxes:
        return None, None

    # 面積が最大の顔を選ぶ
    largest = max(boxes, key=lambda b: b.w * b.h)
    (face_img,) = crop_faces(image, [largest])

    return largest, face_img


if __name__ == "__main__":
    """
    簡単な動作確認用。

    例:
        python -m src.inference.face_detector

    実行前に、image_path を手元の画像ファイルに変更してください。
    """
    import os

    # テスト用のパス（適宜書き換えてください）
    image_path = Path("tests/face.jpg") # 例: "tests/sample_image.jpg"

    if image_path is None or not os.path.exists(image_path):
        print("[WARN] テスト用画像パスを face_detector.py 内で設定してください。")
    else:
        img = cv2.imread(image_path)  # BGR で読み込まれる
        boxes = detect_faces(img, bgr=True)
        print(f"Detected {len(boxes)} face(s).")
        for i, box in enumerate(boxes):
            print(f"Face {i}: {box}")

        # 最も大きい顔だけテスト
        largest_box, face_img = detect_and_crop_largest_face(img, bgr=True)
        if largest_box is not None:
            print("Largest face:", largest_box)
            # テストとして保存
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite("outputs/cropped_face.jpg", face_img)
            print("Saved cropped face to outputs/cropped_face.jpg")
