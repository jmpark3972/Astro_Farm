#!/usr/bin/env python3
"""센서를 하나씩 읽어 확인합니다 (Raspberry Pi / DietPi).

사용 예:
  python3 sensor_test.py              # 대화형 메뉴
  python3 sensor_test.py dht11
  python3 sensor_test.py dht11                      # 기본 BCM GPIO4
  python3 sensor_test.py --dht-bcm 22 dht11        # 다른 BCM으로 테스트
  ASTROFARM_DHT_BCM=22 python3 sensor_test.py dht11
  python3 sensor_test.py as7262
  python3 sensor_test.py ads1015
  python3 sensor_test.py all
  python3 sensor_test.py --mock dht11 # PC 등 하드웨어 없이 값 형식만 확인
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def _print_block(title: str, data: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _mock_slice(full: Dict[str, Any], sensor: str) -> Dict[str, Any]:
    if sensor == "dht11":
        return {
            "_probe": "DHT11",
            "_hardware": False,
            "temp_air": full.get("temp_air"),
            "humidity": full.get("humidity"),
        }
    if sensor == "as7262":
        keys = [k for k in full if k.startswith("par_")]
        out = {k: full[k] for k in sorted(keys)}
        out["_probe"] = "AS7262"
        out["_hardware"] = False
        return out
    if sensor == "ads1015":
        return {
            "_probe": "ADS1015",
            "_hardware": False,
            "ec": full.get("ec"),
            "ph": full.get("ph"),
            "co2": full.get("co2"),
            "water_temp": full.get("water_temp"),
        }
    raise ValueError(sensor)


def run_probe(sensor: str, *, use_mock: bool) -> None:
    from astrofarm_fixed import HARDWARE, MockSensorManager, SensorManager

    if use_mock:
        mgr = MockSensorManager()
        full = mgr.read_once()
        if sensor == "all":
            _print_block("DHT11", _mock_slice(full, "dht11"))
            _print_block("AS7262", _mock_slice(full, "as7262"))
            _print_block("ADS1015", _mock_slice(full, "ads1015"))
        else:
            _print_block(sensor.upper(), _mock_slice(full, sensor))
        return

    mgr = SensorManager()
    if sensor == "dht11":
        _print_block("DHT11", mgr.probe_dht11())
    elif sensor == "as7262":
        _print_block("AS7262", mgr.probe_as7262())
    elif sensor == "ads1015":
        _print_block("ADS1015", mgr.probe_ads1015())
    elif sensor == "all":
        _print_block("DHT11", mgr.probe_dht11())
        _print_block("AS7262", mgr.probe_as7262())
        _print_block("ADS1015", mgr.probe_ads1015())
    else:
        raise ValueError(sensor)


def _menu() -> None:
    from astrofarm_fixed import DHT11_BCM_PIN, HARDWARE

    print(
        "센서 단독 테스트 — 번호 선택 후 Enter (종료: q)\n"
        f"  1 : DHT11 (BCM GPIO{DHT11_BCM_PIN})\n"
        "  2 : AS7262 (I2C 0x49)\n"
        "  3 : ADS1015 (I2C 0x48, A0~A3)\n"
        "  4 : 위 세 가지 연속\n"
        "  m : MockSensorManager (시뮬 값)\n"
        "  q : 종료"
    )
    use_mock = False
    while True:
        try:
            line = input("\n선택> ").strip().lower()
        except EOFError:
            print()
            break
        if line in ("q", "quit", ""):
            break
        if line == "m":
            use_mock = not use_mock
            print(f"[모드] mock={'ON' if use_mock else 'OFF'}, HARDWARE={HARDWARE}")
            continue
        choice = {"1": "dht11", "2": "as7262", "3": "ads1015", "4": "all"}.get(line)
        if not choice:
            print("1~4, m, q 중에서 입력하세요.")
            continue
        try:
            run_probe(choice, use_mock=use_mock)
        except Exception as e:
            print(f"오류: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="센서 단독 읽기 테스트")
    parser.add_argument(
        "--dht-bcm",
        type=int,
        default=None,
        metavar="N",
        help="DHT11 BCM GPIO 번호 (미지정 시 환경변수 ASTROFARM_DHT_BCM 또는 기본 4)",
    )
    parser.add_argument(
        "sensor",
        nargs="?",
        choices=("dht11", "as7262", "ads1015", "all"),
        help="테스트할 센서 (생략 시 대화형 메뉴)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="MockSensorManager로 시뮬 값만 출력",
    )
    args = parser.parse_args()

    if args.dht_bcm is not None:
        os.environ["ASTROFARM_DHT_BCM"] = str(args.dht_bcm)

    if args.sensor is None:
        _menu()
        return 0

    try:
        run_probe(args.sensor, use_mock=args.mock)
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
