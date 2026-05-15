# Astro_Farm

라즈베리파이 기반 스마트팜 통합 제어 프로젝트입니다.  
센서 수집, 온도 제어(PID), 식물 영상 지수(ExG), LSTM 이상탐지, XBee 원격 통신을 포함합니다.

## 주요 구성

- `astrofarm_fixed.py`
  - `SensorManager` / `MockSensorManager`
  - `TemperatureController` (PID + 릴레이 방향 제어)
  - `ExGAnalyzer` (Pi Camera V2 + Otsu)
  - `XBeeComm` (텔레메트리/텔레커맨드/시퀀스 손실 감지)
- `train_lstm_anomaly.py`
  - 지상국 PC(TensorFlow) 학습
- `convert_lstm_to_tflite.py`
  - Keras -> TFLite 변환
- `infer_tflite_pi.py`
  - Pi Zero용 TFLite 추론
- `main.py`
  - 통합 실행 루프(5초 주기)

## 실행 순서 (main.py)

1. `SensorManager`
2. `TemperatureController`
3. `LSTMInference`
4. `ExGAnalyzer`
5. `XBeeComm`

각 모듈 실패 시 해당 모듈만 스킵하고 루프는 계속 동작합니다.

## 모터(서보) 기준값

영양액 밸브 모터 제어 기준:

- **600일 때 닫힘**
- **1500일 때 열림**

운용 시 해당 기준을 제어 로직/명령값(텔레커맨드)과 일치시키세요.

## 빠른 실행

```bash
python main.py
```

주요 옵션 예시:

```bash
python main.py --uart-port /dev/ttyS0 --baudrate 38400 --target-temp 22
```

## 테스트/시뮬레이션

- 하드웨어 없이 테스트할 때는 `MockSensorManager` 사용
- 단위 테스트 실행:

```bash
python astrofarm_fixed.py --mode test
```

## LSTM 학습 데이터

- `astrofarm_fixed.py --mode run` 은 기본적으로 `astrofarm_data.jsonl` 에 한 줄씩 기록합니다.
- `main.py` 는 `logs/astrofarm_YYYYMMDD.csv` 에 기록합니다.
- `train_lstm_anomaly.py` 는 `--data` 를 비우면 **같은 폴더의 `astrofarm_data.jsonl`** 과 **`logs/` 안 최신 CSV** 중 수정 시각이 더 최근인 파일을 자동으로 고릅니다.
- 슬라이딩 윈도우 `W=60` 이면 **최소 61행 이상** 필요합니다.

```bash
python3 train_lstm_anomaly.py
python3 train_lstm_anomaly.py --data logs/astrofarm_20260515.csv
```
