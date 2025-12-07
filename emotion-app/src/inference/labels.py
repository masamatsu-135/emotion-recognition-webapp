# labels.py

"""
FER-2013 の感情クラスラベルを定義するモジュール。
推論時にモデルの出力インデックスを人間向けのラベルへ変換するために使用。

FER-2013 のクラス一覧:
0 Angry
1 Disgust
2 Fear
3 Happy
4 Sad
5 Surprise
6 Neutral
"""

# 英語ラベル（モデル出力のインデックスに対応）
EMOTION_LABELS_EN = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral",   # 6
]

# 日本語ラベル（UI表示用）
EMOTION_LABELS_JA = [
    "怒り",        # 0
    "嫌悪",        # 1
    "恐れ",        # 2
    "幸福",        # 3
    "悲しみ",      # 4
    "驚き",        # 5
    "ニュートラル"  # 6
]

def get_label_en(class_id: int) -> str:
    """クラスID（0〜6）から英語ラベルを取得"""
    return EMOTION_LABELS_EN[class_id]

def get_label_ja(class_id: int) -> str:
    """クラスID（0〜6）から日本語ラベルを取得"""
    return EMOTION_LABELS_JA[class_id]
