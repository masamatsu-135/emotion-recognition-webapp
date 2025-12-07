# app.py

from __future__ import annotations

from pathlib import Path
from typing import Literal
import sys

import streamlit as st
import numpy as np
import cv2
from PIL import Image

#from ..inference.predictor import EmotionPredictor
#from ..inference.face_detector import detect_and_crop_largest_face
#from ..inference.labels import EMOTION_LABELS_EN, EMOTION_LABELS_JA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import EmotionPredictor
from src.inference.face_detector import detect_and_crop_largest_face
from src.inference.labels import EMOTION_LABELS_EN, EMOTION_LABELS_JA

ModelType = Literal["cnn", "resnet"]

# プロジェクトルート（emotion-app）を __file__ から推定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


@st.cache_resource
def load_predictor(model_type: ModelType) -> EmotionPredictor:
    """
    モデル種別ごとに EmotionPredictor をキャッシュしておく。
    Streamlit の再実行でも毎回ロードし直さなくて済む。
    """
    if model_type == "cnn":
        ckpt = CHECKPOINT_DIR / "best_cnn_fer2013.pth"
    else:
        ckpt = CHECKPOINT_DIR / "best_resnet_fer2013.pth"

    predictor = EmotionPredictor(
        model_type=model_type,
        checkpoint_path=str(ckpt),
        device=None,  # GPUあればGPU、なければCPU
    )
    return predictor


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """
    PIL.Image (RGB) -> OpenCV形式 BGR np.ndarray に変換
    """
    rgb = np.array(image)
    bgr = rgb[:, :, ::-1]
    return bgr


def draw_box_and_label(
    image_bgr: np.ndarray,
    box,
    label_ja: str,
    score: float,
) -> np.ndarray:
    """
    BGR画像に、顔の枠とラベルを描画して返す。
    """
    annotated = image_bgr.copy()
    x, y, w, h = box.as_tuple()

    # 顔の枠
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ラベルテキスト
    text = f"{label_ja} ({score:.2f})"
    cv2.putText(
        annotated,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated


def main():
    st.set_page_config(page_title="表情認識デモ", page_icon="🙂")
    st.title("表情認識アプリ（FER-2013）")

    st.write("アップロード画像 or Webカメラから顔の感情を推定します。")

    # ==== サイドバー ====
    st.sidebar.header("設定")

    model_type: ModelType = st.sidebar.selectbox(
        "モデルタイプ",
        options=["cnn", "resnet"],
        format_func=lambda x: "CNN（軽量）" if x == "cnn" else "ResNet-18（高精度）",
    )

    st.sidebar.write("※ ResNet は学習にGPUを使いましたが、推論はCPUでも動作します。")

    # 予測実行ボタンが押されたときだけモデルをロードしたいので、先にボタン定義
    input_mode = st.radio(
        "入力方法を選択してください",
        options=["画像アップロード", "Webカメラ"],
        horizontal=True,
    )

    uploaded_image = None

    if input_mode == "画像アップロード":
        file = st.file_uploader("顔が写った画像をアップロードしてください", type=["jpg", "jpeg", "png"])
        if file is not None:
            uploaded_image = Image.open(file).convert("RGB")
            st.image(uploaded_image, caption="アップロード画像", use_column_width=True)
    else:
        camera_image = st.camera_input("Webカメラで撮影")
        if camera_image is not None:
            uploaded_image = Image.open(camera_image).convert("RGB")
            st.image(uploaded_image, caption="撮影画像", use_column_width=True)

    run_button = st.button("表情を推定する")

    if run_button:
        if uploaded_image is None:
            st.warning("先に画像を用意してください。")
            return

        # ===== 1. モデル読み込み =====
        try:
            predictor = load_predictor(model_type)
        except FileNotFoundError as e:
            st.error(
                f"モデルファイルが見つかりませんでした。\n{e}\n"
                "train.py で学習を行い、.pth を models/checkpoints/ に配置してください。"
            )
            return

        # ===== 2. 顔検出 =====
        bgr_image = pil_to_bgr(uploaded_image)
        box, face_img = detect_and_crop_largest_face(bgr_image, bgr=True)

        if face_img is None:
            st.error("顔が検出できませんでした。別の画像で試してみてください。")
            return

        # ===== 3. 感情推定 =====
        result = predictor.predict_from_ndarray(face_img, bgr=True)
        class_id = result["class_id"]
        label_ja = result["label_ja"]
        label_en = result["label_en"]
        probs = result["probs"]
        confidence = probs[class_id]

        st.subheader("推定結果")
        st.markdown(
            f"**予測された感情:** {label_ja}（{label_en}）  \n"
            f"**確信度:** {confidence:.2%}"
        )

        # ===== 4. 全クラスの確率を可視化 =====
        st.write("各感情クラスの確率:")
        prob_dict = {
            f"{EMOTION_LABELS_JA[i]} ({EMOTION_LABELS_EN[i]})": probs[i]
            for i in range(len(probs))
        }
        st.bar_chart(prob_dict)

        # ===== 5. 顔の枠を描画した画像を表示 =====
        annotated_bgr = draw_box_and_label(bgr_image, box, label_ja, confidence)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        st.image(annotated_rgb, caption="検出された顔と推定結果", use_column_width=True)


if __name__ == "__main__":
    main()
