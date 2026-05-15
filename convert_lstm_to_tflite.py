"""
AstroFarm LSTM 모델 -> TFLite 변환 스크립트
"""

from __future__ import annotations

import argparse
import os

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="Keras LSTM 모델을 TFLite로 변환")
    parser.add_argument("--model", type=str, default="models/lstm_anomaly.keras")
    parser.add_argument("--out", type=str, default="models/lstm_anomaly.tflite")
    parser.add_argument("--float16", action="store_true", help="float16 양자화 사용")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(args.out, "wb") as f:
        f.write(tflite_model)

    print("TFLite 변환 완료")
    print(f"- input model: {args.model}")
    print(f"- output file: {args.out}")
    print(f"- size: {len(tflite_model) / 1024:.2f} KB")


if __name__ == "__main__":
    main()

