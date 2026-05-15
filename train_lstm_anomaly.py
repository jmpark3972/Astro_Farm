"""
AstroFarm LSTM 이상탐지 모델 학습 스크립트 (지상국 PC, TensorFlow)

요구사항:
- 입력 변수 7개: [temperature, humidity, co2, ec, ph, par, exg]
- 슬라이딩 윈도우: W=60
- 모델: LSTM(64) -> LSTM(32) -> Dense(16, relu) -> Dense(7)
- 손실함수: MSE, 옵티마이저: Adam
- 이상 탐지 기준: 예측값 vs 실제값 MAE > threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


FEATURE_NAMES = ["temperature", "humidity", "co2", "ec", "ph", "par", "exg"]


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def extract_feature_vector(row: Dict) -> np.ndarray:
    """
    로거/SensorManager 두 포맷을 모두 허용하여 7개 특징 벡터 추출.
    """
    temperature = _safe_float(row.get("temperature", row.get("temp_air")))
    humidity = _safe_float(row.get("humidity"))
    co2 = _safe_float(row.get("co2_ppm", row.get("co2")))
    ec = _safe_float(row.get("ec_ms_cm", row.get("ec")))
    ph = _safe_float(row.get("ph"))
    par = _safe_float(row.get("par_ue", row.get("par")))
    exg = _safe_float(row.get("exg_mean", row.get("exg")))

    return np.array([temperature, humidity, co2, ec, ph, par, exg], dtype=np.float32)


def load_rows(data_path: str) -> List[Dict]:
    ext = os.path.splitext(data_path.lower())[1]
    rows: List[Dict] = []

    if ext == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if ext == ".csv":
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
        return rows

    raise ValueError("지원하지 않는 데이터 형식입니다. (.jsonl 또는 .csv 사용)")


def rows_to_feature_matrix(rows: List[Dict]) -> np.ndarray:
    data = np.array([extract_feature_vector(r) for r in rows], dtype=np.float32)
    if data.size == 0:
        return data

    # 결측치 보간: 각 feature의 중앙값으로 대체
    med = np.nanmedian(data, axis=0)
    nan_idx = np.isnan(data)
    data[nan_idx] = np.take(med, np.where(nan_idx)[1])
    return data


def make_sliding_windows(data: np.ndarray, window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    X[t] = data[t:t+W], y[t] = data[t+W]
    """
    if len(data) <= window:
        raise ValueError(
            f"데이터가 부족합니다. 최소 {window + 1}개 샘플 필요, 현재 {len(data)}개"
        )

    xs = []
    ys = []
    for i in range(len(data) - window):
        xs.append(data[i : i + window])
        ys.append(data[i + window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_model(window: int = 60, n_features: int = 7) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(window, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(n_features)(x)
    model = tf.keras.Model(inp, out, name="astrofarm_lstm_anomaly")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


def main():
    parser = argparse.ArgumentParser(description="AstroFarm LSTM 이상탐지 학습")
    parser.add_argument("--data", type=str, default="astrofarm_data.jsonl")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--model-out", type=str, default="models/lstm_anomaly.keras")
    parser.add_argument("--scaler-out", type=str, default="models/lstm_scaler.npz")
    parser.add_argument("--meta-out", type=str, default="models/lstm_meta.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.scaler_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)

    rows = load_rows(args.data)
    data = rows_to_feature_matrix(rows)
    x_raw, y_raw = make_sliding_windows(data, window=args.window)

    # 정규화 파라미터는 학습 데이터 전체 기반
    feature_mean = data.mean(axis=0)
    feature_std = data.std(axis=0) + 1e-6
    x = (x_raw - feature_mean) / feature_std
    y = (y_raw - feature_mean) / feature_std

    model = build_model(window=args.window, n_features=len(FEATURE_NAMES))
    history = model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        shuffle=False,
        verbose=1,
    )

    # MAE 임계값 계산 (학습 데이터 기반)
    pred_norm = model.predict(x, verbose=0)
    pred_raw = (pred_norm * feature_std) + feature_mean
    mae_per_sample = np.mean(np.abs(pred_raw - y_raw), axis=1)
    threshold_mae = float(np.percentile(mae_per_sample, 95))

    model.save(args.model_out)
    np.savez(args.scaler_out, mean=feature_mean, std=feature_std)

    meta = {
        "feature_names": FEATURE_NAMES,
        "window": int(args.window),
        "threshold_mae": threshold_mae,
        "train_samples": int(len(x)),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("학습 완료")
    print(f"- model:   {args.model_out}")
    print(f"- scaler:  {args.scaler_out}")
    print(f"- meta:    {args.meta_out}")
    print(f"- threshold_mae (95p): {threshold_mae:.6f}")


if __name__ == "__main__":
    main()

