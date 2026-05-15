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
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


FEATURE_NAMES = ["temperature", "humidity", "co2", "ec", "ph", "par", "exg"]

# main.py CSV 의 PAR 채널 → astrofarm_fixed.PAR_WEIGHTS 와 동일 가중치
CSV_PAR_BAND_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("sensor_par_450", 0.10),
    ("sensor_par_500", 0.15),
    ("sensor_par_550", 0.20),
    ("sensor_par_570", 0.20),
    ("sensor_par_600", 0.15),
    ("sensor_par_650", 0.20),
)


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, str) and not str(v).strip():
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _pick(row: Dict, *keys: str):
    """CSV/JSON 행에서 첫 번째 비어 있지 않은 값 선택 (0.0도 유효)."""
    for k in keys:
        if k not in row:
            continue
        v = row[k]
        if v is None:
            continue
        if isinstance(v, str) and not str(v).strip():
            continue
        return v
    return None


def _weighted_par_from_main_csv(row: Dict) -> float:
    total_w = 0.0
    acc = 0.0
    for key, w in CSV_PAR_BAND_WEIGHTS:
        val = _safe_float(_pick(row, key))
        if not np.isnan(val):
            acc += val * w
            total_w += w
    if total_w <= 0:
        return float(np.nan)
    return float(acc / total_w)


def extract_feature_vector(row: Dict) -> np.ndarray:
    """
    로거/SensorManager/main.py CSV 포맷을 허용하여 7개 특징 벡터 추출.
    """
    temperature = _safe_float(_pick(row, "temperature", "temp_air", "sensor_temp_air"))
    humidity = _safe_float(_pick(row, "humidity", "sensor_humidity"))
    co2 = _safe_float(_pick(row, "co2_ppm", "co2", "sensor_co2"))
    ec = _safe_float(_pick(row, "ec_ms_cm", "ec", "sensor_ec"))
    ph = _safe_float(_pick(row, "ph", "sensor_ph"))
    par = _safe_float(_pick(row, "par_ue", "par"))
    if np.isnan(par):
        par = _weighted_par_from_main_csv(row)

    exg = _safe_float(_pick(row, "exg_mean", "exg", "exg_mean_exg"))

    return np.array([temperature, humidity, co2, ec, ph, par, exg], dtype=np.float32)


def resolve_training_data_path(cli_path: str) -> str:
    """
    --data 가 비어 있으면 astrofarm_data.jsonl 또는 logs/astrofarm_*.csv 중
    수정 시각이 가장 최근인 파일을 선택합니다.
    """
    cli_path = (cli_path or "").strip()
    if cli_path:
        if os.path.isfile(cli_path):
            return os.path.abspath(cli_path)
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {cli_path}\n\n"
            "준비 방법:\n"
            "  • 보드에서 python3 main.py 를 충분히 실행해 logs/astrofarm_YYYYMMDD.csv 생성\n"
            "  • 또는 astrofarm_fixed.py --mode run 으로 astrofarm_data.jsonl 생성\n"
            "  • 또는 명시 경로: python3 train_lstm_anomaly.py --data logs/astrofarm_20260515.csv"
        )

    candidates: List[Tuple[float, str]] = []
    if os.path.isfile("astrofarm_data.jsonl"):
        candidates.append((os.path.getmtime("astrofarm_data.jsonl"), "astrofarm_data.jsonl"))
    for p in glob.glob(os.path.join("logs", "astrofarm_*.csv")):
        candidates.append((os.path.getmtime(p), p))

    if not candidates:
        raise FileNotFoundError(
            "학습용 데이터 파일이 없습니다.\n\n"
            "먼저 장비에서 main.py 로 로그를 쌓거나,\n"
            "astrofarm_data.jsonl 을 이 디렉터리에 두고 다시 실행하세요.\n"
            "(윈도우가 W=60이면 최소 61개 행 이상 필요)"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return os.path.abspath(candidates[0][1])


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

    # 결측치 보간: 각 feature의 중앙값으로 대체 (컬럼 전체 NaN이면 0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(data, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
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
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="학습 데이터(.jsonl 또는 .csv). 비우면 astrofarm_data.jsonl 또는 logs/ 최신 CSV 자동 선택",
    )
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

    data_path = resolve_training_data_path(args.data)
    print(f"[train_lstm_anomaly] 데이터: {data_path}")
    rows = load_rows(data_path)
    n_rows = len(rows)
    need = args.window + 1
    if n_rows < need:
        raise SystemExit(
            f"학습 데이터 행 수 부족: 현재 {n_rows}행, 최소 {need}행 필요 (--window={args.window}).\n"
            f"  • main.py 를 더 오래 실행해 CSV를 쌓거나\n"
            f"  • 테스트용으로 --window {max(1, n_rows - 1)} 로 줄여 보세요."
        )

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

