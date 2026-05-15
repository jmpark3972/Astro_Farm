"""
AstroFarm 통합 실행 엔트리포인트

실행 순서 (요구사항):
  SensorManager -> TemperatureController -> LSTMInference -> ExGAnalyzer -> XBeeComm

특징:
  - 60초 주기 메인 루프
  - 모듈 단위 장애 격리(실패 모듈만 스킵, 루프 지속)
  - 날짜별 CSV 로컬 저장
  - systemd 운용을 위한 SIGTERM/SIGINT 핸들링
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import time
from datetime import datetime
from threading import Event
from typing import Any, Dict, Optional

from astrofarm_fixed import (
    ExGAnalyzer,
    SensorManager,
    TemperatureController,
    XBeeComm,
)
from infer_tflite_pi import TFLiteAnomalyDetector


log = logging.getLogger("AstroFarmMain")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


class LSTMInference:
    """Pi Zero TFLite 추론 래퍼. 초기화 실패 시 비활성 모드로 동작."""

    def __init__(self, model_path: str, scaler_path: str, meta_path: str):
        self._detector: Optional[TFLiteAnomalyDetector] = None
        self._enabled = False
        try:
            self._detector = TFLiteAnomalyDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                meta_path=meta_path,
            )
            self._enabled = True
            log.info("LSTMInference 초기화 완료")
        except Exception as e:
            log.error("LSTMInference 초기화 실패: %s", e)
            self._enabled = False

    def detect(self, sensor_row: Dict[str, Any]) -> Dict[str, Any]:
        if not self._enabled or self._detector is None:
            return {
                "ready": False,
                "warning": False,
                "mae": None,
                "threshold_mae": None,
                "pred": None,
                "actual": None,
                "skipped": True,
            }
        return self._detector.detect(sensor_row)


class DailyCSVLogger:
    """날짜별 CSV 파일 분리 저장."""

    FIELDNAMES = [
        "timestamp",
        "sensor_temp_air",
        "sensor_humidity",
        "sensor_co2",
        "sensor_ec",
        "sensor_ph",
        "sensor_water_temp",
        "sensor_par_450",
        "sensor_par_500",
        "sensor_par_550",
        "sensor_par_570",
        "sensor_par_600",
        "sensor_par_650",
        "control_current_temp",
        "control_error",
        "control_pid_output",
        "control_relay_on",
        "control_relay_mode",
        "control_target_temp",
        "lstm_ready",
        "lstm_warning",
        "lstm_mae",
        "lstm_threshold_mae",
        "exg_coverage_pct",
        "exg_mean_exg",
        "exg_drop_warning",
        "exg_drop_pct",
        "xbee_tx_ok",
        "module_errors",
    ]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _file_path(self, now: datetime) -> str:
        return os.path.join(self.log_dir, f"astrofarm_{now.strftime('%Y%m%d')}.csv")

    def write(self, row: Dict[str, Any]):
        now = datetime.now()
        path = self._file_path(now)
        need_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if need_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.FIELDNAMES})


class AstroFarmMain:
    LOOP_SEC = 60.0

    def __init__(self, args):
        self.args = args
        self.stop_event = Event()
        self.last_exg_mean = 0.0

        self.sensor_mgr = SensorManager(period_sec=5.0)
        self.temp_ctrl = TemperatureController(
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
            target_temp=args.target_temp,
            control_period_sec=10.0,
        )
        self.lstm = LSTMInference(
            model_path=args.model,
            scaler_path=args.scaler,
            meta_path=args.meta,
        )
        self.exg = ExGAnalyzer(out_dir=args.capture_dir)
        self.xbee = XBeeComm(
            port=args.uart_port,
            baudrate=args.baudrate,
            one_way_delay_sec=1.28,
        )
        self.csv_logger = DailyCSVLogger(log_dir=args.log_dir)

    def _handle_signal(self, signum, _frame):
        log.info("종료 시그널 수신: %s", signum)
        self.stop_event.set()

    def install_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    @staticmethod
    def _sanitize_exg_for_tx(exg_result: Dict[str, Any]) -> Dict[str, Any]:
        # ndarray(exg_map)는 패킷/CSV에 그대로 실지 않음
        if not exg_result:
            return {}
        out = dict(exg_result)
        out.pop("exg_map", None)
        return out

    def run(self):
        log.info("AstroFarmMain 시작 (loop=60s)")
        while not self.stop_event.is_set():
            t0 = time.time()
            timestamp = datetime.now().isoformat()
            module_errors: Dict[str, str] = {}

            sensor_data: Dict[str, Any] = {}
            control_state: Dict[str, Any] = {}
            lstm_pred: Dict[str, Any] = {}
            exg_result: Dict[str, Any] = {}
            xbee_tx_ok = False

            # 1) SensorManager
            try:
                sensor_data = self.sensor_mgr.read_once()
            except Exception as e:
                module_errors["SensorManager"] = str(e)
                log.error("SensorManager 실패: %s", e)

            # 2) TemperatureController
            try:
                if sensor_data:
                    control_state = self.temp_ctrl.control_once(sensor_data)
                else:
                    raise RuntimeError("sensor_data unavailable")
            except Exception as e:
                module_errors["TemperatureController"] = str(e)
                log.error("TemperatureController 실패: %s", e)
                control_state = {}

            # 3) LSTMInference
            try:
                if sensor_data:
                    lstm_input = dict(sensor_data)
                    # 요구 피처 7개 중 ExG는 직전 관측값을 사용 (순서 보장 목적)
                    lstm_input["exg"] = self.last_exg_mean
                    lstm_pred = self.lstm.detect(lstm_input)
                else:
                    raise RuntimeError("sensor_data unavailable")
            except Exception as e:
                module_errors["LSTMInference"] = str(e)
                log.error("LSTMInference 실패: %s", e)
                lstm_pred = {}

            # 4) ExGAnalyzer
            try:
                exg_result = self.exg.capture_and_analyze()
                if exg_result.get("mean_exg") is not None:
                    self.last_exg_mean = float(exg_result["mean_exg"])
            except Exception as e:
                module_errors["ExGAnalyzer"] = str(e)
                log.error("ExGAnalyzer 실패: %s", e)
                exg_result = {}

            # 5) XBeeComm
            try:
                xbee_tx_ok = self.xbee.send_telemetry(
                    sensor_data=sensor_data,
                    control_state=control_state,
                    lstm_prediction=lstm_pred,
                    exg_result=self._sanitize_exg_for_tx(exg_result),
                )
            except Exception as e:
                module_errors["XBeeComm"] = str(e)
                log.error("XBeeComm 실패: %s", e)
                xbee_tx_ok = False

            row = {
                "timestamp": timestamp,
                "sensor_temp_air": sensor_data.get("temp_air"),
                "sensor_humidity": sensor_data.get("humidity"),
                "sensor_co2": sensor_data.get("co2"),
                "sensor_ec": sensor_data.get("ec"),
                "sensor_ph": sensor_data.get("ph"),
                "sensor_water_temp": sensor_data.get("water_temp"),
                "sensor_par_450": sensor_data.get("par_450"),
                "sensor_par_500": sensor_data.get("par_500"),
                "sensor_par_550": sensor_data.get("par_550"),
                "sensor_par_570": sensor_data.get("par_570"),
                "sensor_par_600": sensor_data.get("par_600"),
                "sensor_par_650": sensor_data.get("par_650"),
                "control_current_temp": control_state.get("current_temp"),
                "control_error": control_state.get("error"),
                "control_pid_output": control_state.get("pid_output"),
                "control_relay_on": control_state.get("relay_on"),
                "control_relay_mode": control_state.get("relay_mode"),
                "control_target_temp": control_state.get("target_temp"),
                "lstm_ready": lstm_pred.get("ready"),
                "lstm_warning": lstm_pred.get("warning"),
                "lstm_mae": lstm_pred.get("mae"),
                "lstm_threshold_mae": lstm_pred.get("threshold_mae"),
                "exg_coverage_pct": exg_result.get("coverage_pct"),
                "exg_mean_exg": exg_result.get("mean_exg"),
                "exg_drop_warning": exg_result.get("drop_warning"),
                "exg_drop_pct": exg_result.get("drop_pct"),
                "xbee_tx_ok": xbee_tx_ok,
                "module_errors": json.dumps(module_errors, ensure_ascii=False),
            }

            try:
                self.csv_logger.write(row)
            except Exception as e:
                # 로거 실패도 루프 지속
                log.error("CSV 로깅 실패: %s", e)

            elapsed = time.time() - t0
            sleep_sec = max(0.0, self.LOOP_SEC - elapsed)
            self.stop_event.wait(timeout=sleep_sec)

        self.cleanup()

    def cleanup(self):
        log.info("AstroFarmMain 정리 시작")
        try:
            self.temp_ctrl.cleanup()
        except Exception:
            pass
        try:
            self.sensor_mgr.close()
        except Exception:
            pass
        try:
            self.exg.stop_daily_schedule()
        except Exception:
            pass
        try:
            self.xbee.close()
        except Exception:
            pass
        log.info("AstroFarmMain 종료")


def parse_args():
    parser = argparse.ArgumentParser(description="AstroFarm 통합 메인 루프")
    parser.add_argument("--uart-port", type=str, default="/dev/ttyS0")
    parser.add_argument("--baudrate", type=int, default=38400)
    parser.add_argument("--target-temp", type=float, default=22.0)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="models/lstm_anomaly.tflite")
    parser.add_argument("--scaler", type=str, default="models/lstm_scaler.npz")
    parser.add_argument("--meta", type=str, default="models/lstm_meta.json")
    parser.add_argument("--capture-dir", type=str, default="captures")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def main():
    args = parse_args()
    app = AstroFarmMain(args)
    app.install_signal_handlers()
    app.run()


if __name__ == "__main__":
    main()

