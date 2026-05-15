"""
AstroFarm Pi Zero 2W용 TFLite 추론 (이상 탐지)

사용 데이터 형식:
  x(t) = [온도, 습도, CO2, EC, pH, PAR, ExG]
이상 조건:
  MAE(pred, actual) > threshold_mae -> warning=True
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

Interpreter = None
INTERPRETER_BACKEND = "none"

try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter

    Interpreter = _TFLiteInterpreter
    INTERPRETER_BACKEND = "tflite_runtime"
except Exception:
    try:
        # 개발 PC fallback
        import tensorflow as tf

        _lite = getattr(tf, "lite", None)
        if _lite is not None and hasattr(_lite, "Interpreter"):
            Interpreter = _lite.Interpreter
            INTERPRETER_BACKEND = "tensorflow.lite"
    except Exception:
        Interpreter = None
        INTERPRETER_BACKEND = "none"


FEATURE_NAMES = ["temperature", "humidity", "co2", "ec", "ph", "par", "exg"]


def _tflite_install_hint() -> str:
    """pip 에 wheel 이 없을 때(특히 Py 3.13 + aarch64) 안내."""
    parts = [
        "pip install tflite-runtime",
        "또는 pip install tensorflow(tf.lite).",
    ]
    if sys.version_info >= (3, 13):
        parts.append(
            "Raspberry Pi 에서 Python 3.13 은 tflite-runtime 공식 wheel 이 없을 수 있습니다. "
            "apt 로 python3.11·python3.11-venv 설치 후 python3.11 -m venv 로 새 환경을 만들고 "
            "그 안에서 pip install tflite-runtime 하세요."
        )
    return " ".join(parts)


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def extract_feature_vector(row: Dict[str, Any]) -> np.ndarray:
    temperature = _safe_float(row.get("temperature", row.get("temp_air")))
    humidity = _safe_float(row.get("humidity"))
    co2 = _safe_float(row.get("co2_ppm", row.get("co2")))
    ec = _safe_float(row.get("ec_ms_cm", row.get("ec")))
    ph = _safe_float(row.get("ph"))
    par = _safe_float(row.get("par_ue", row.get("par")))
    exg = _safe_float(row.get("exg_mean", row.get("exg")))

    vec = np.array([temperature, humidity, co2, ec, ph, par, exg], dtype=np.float32)
    # 결측치 대비: 0 대체 (권장: 호출부에서 결측 제거)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec


class TFLiteAnomalyDetector:
    def __init__(self, model_path: str, scaler_path: str, meta_path: str):
        if Interpreter is None:
            raise RuntimeError(
                "TFLite interpreter backend not available. " + _tflite_install_hint()
            )

        scaler = np.load(scaler_path)
        self.mean = scaler["mean"].astype(np.float32)
        self.std = scaler["std"].astype(np.float32)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.window = int(meta["window"])
        self.threshold_mae = float(meta["threshold_mae"])

        self.history = deque(maxlen=self.window)

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def _denormalize(self, z: np.ndarray) -> np.ndarray:
        return (z * (self.std + 1e-6)) + self.mean

    def detect(self, current_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        새 관측치 1개를 받아 이상 여부 반환.
        warmup 동안은 warning=False 반환.
        """
        actual_raw = extract_feature_vector(current_row)
        actual_norm = self._normalize(actual_raw)

        # 아직 윈도우가 안 찼으면 적재만 수행
        if len(self.history) < self.window:
            self.history.append(actual_norm)
            return {
                "ready": False,
                "warning": False,
                "mae": None,
                "threshold_mae": self.threshold_mae,
                "pred": None,
                "actual": actual_raw.tolist(),
            }

        x_input = np.array(self.history, dtype=np.float32)[None, :, :]  # (1, W, 7)
        self.interpreter.set_tensor(self.input_details[0]["index"], x_input.astype(np.float32))
        self.interpreter.invoke()
        pred_norm = self.interpreter.get_tensor(self.output_details[0]["index"])[0]  # (7,)
        pred_raw = self._denormalize(pred_norm)

        mae = float(np.mean(np.abs(pred_raw - actual_raw)))
        warning = mae > self.threshold_mae

        # 현재 관측치를 다음 추론 윈도우에 편입
        self.history.append(actual_norm)

        return {
            "ready": True,
            "warning": warning,
            "mae": mae,
            "threshold_mae": self.threshold_mae,
            "pred": pred_raw.tolist(),
            "actual": actual_raw.tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description="Pi Zero TFLite LSTM 이상탐지 추론")
    parser.add_argument("--model", type=str, default="models/lstm_anomaly.tflite")
    parser.add_argument("--scaler", type=str, default="models/lstm_scaler.npz")
    parser.add_argument("--meta", type=str, default="models/lstm_meta.json")
    parser.add_argument(
        "--demo-json",
        type=str,
        default="",
        help="테스트용 1개 샘플 JSON 문자열. 예: '{\"temp_air\":24.3,...}'",
    )
    args = parser.parse_args()

    detector = TFLiteAnomalyDetector(
        model_path=args.model,
        scaler_path=args.scaler,
        meta_path=args.meta,
    )

    if args.demo_json:
        sample = json.loads(args.demo_json)
        result = detector.detect(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            "TFLiteAnomalyDetector 초기화 완료. "
            "코드에서 detector.detect(sensor_dict)를 반복 호출해 사용하세요."
        )


if __name__ == "__main__":
    main()

