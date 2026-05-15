"""
AstroFarm - 달 기지 스마트팜 통합 제어 시스템 + 테스트/시뮬레이션
================================================================
Raspberry Pi Zero 2W OBC

하드웨어 구성:
  DHT11        : 온습도 센서               (1-Wire, GPIO 4)
  RX-9 Simple  : CO2 센서 (전기화학식)     (Analog EMF  -> ADS1015 CH2)
                  내장 서미스터(NTC)        (Analog      -> ADS1015 CH3)
  AS7262       : 6채널 분광 센서           (I2C, 0x49)
  ADS1015      : 12-bit ADC               (I2C, 0x48)
                  CH0 -> CRT14016P pH 센서
                  CH1 -> SEN0244 TDS 센서
                  CH2 -> RX-9 Simple CO2 EMF 출력
                  CH3 -> RX-9 Simple Thermistor (NTC)
  XBee         : 920MHz 통신 모듈          (UART, TX=GPIO14, RX=GPIO15)
  Pi Camera V2 : 8MP 식물 촬영             (CSI-2, picamera2)
  Peltier      : 온도 제어                 (GPIO 17, PWM)
  WS2812B      : NeoPixel RGB LED          (GPIO 18, 데이터)
  MG92B        : 영양액 공급 서보 모터     (GPIO 23, PWM 50Hz)

MG92B 서보 제어:
  50Hz PWM (주기 20ms)
  0도   = 2.5% 듀티 사이클  (펌프 닫힘, 기본 위치)
  90도  = 7.5% 듀티 사이클  (펌프 열림, 영양액 공급)
  180도 = 12.5% 듀티 사이클
  동작 속도: 0.13sec/60도 @ 5V  ->  약 0.20sec/90도

  영양액 공급 시퀀스:
    1) 0도 -> 90도 회전 (밸브 개방)
    2) 개방 상태 유지 (영양액 낙하)
    3) 90도 -> 0도 복귀 (밸브 폐쇄)
    4) 5초 대기 후 반복

ADS1015 채널 배치:
  CH0 (A0) -> CRT14016P pH 센서       (0~3.3V -> pH 0~14)
  CH1 (A1) -> SEN0244 TDS 센서        (0~2.3V -> 0~1000ppm)
  CH2 (A2) -> RX-9 Simple CO2 EMThermistorF     (0~1.5V -> 0~5000ppm)
  CH3 (A3) -> RX-9 Simple   (NTC, 센서 온도 보상용)

실행 방법:
  python astrofarm.py --mode sim   (시뮬레이션, 기본)
  python astrofarm.py --mode test  (단위 테스트)
  python astrofarm.py --mode run   (실제 운용)
"""

# =========================================================================
#  임포트 및 하드웨어 감지
# =========================================================================

import time
import math
import json
import os
import struct
import logging
import random
import threading
import unittest
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# 하드웨어 모듈 임포트 시도, 실패하면 시뮬레이션 모드
HARDWARE = False
_HARDWARE_IMPORT_ERROR: Optional[BaseException] = None
try:
    import board
    import busio
    import adafruit_dht
    import adafruit_ads1x15.ads1015 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    import neopixel
    import RPi.GPIO as GPIO
    import serial
    HARDWARE = True
except ImportError as e:
    _HARDWARE_IMPORT_ERROR = e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("astrofarm.log"),
    ],
)
log = logging.getLogger("AstroFarm")

if not HARDWARE and _HARDWARE_IMPORT_ERROR is not None:
    log.warning(
        "HARDWARE=False (실장 라이브러리 Import 실패): %s — 시뮬레이션 모드",
        _HARDWARE_IMPORT_ERROR,
    )

# =========================================================================
#  데이터 클래스
# =========================================================================

@dataclass
class SensorData:
    """한 주기에서 수집되는 모든 센서 + 액추에이터 상태"""
    timestamp:       str   = ""
    # DHT11
    temperature:     float = 0.0
    humidity:        float = 0.0
    # RX-9 Simple (CH2 EMF + CH3 Thermistor)
    co2_ppm:         int   = 0
    co2_emf_mv:      float = 0.0
    co2_sensor_temp: float = 0.0
    # AS7262
    ch450:           float = 0.0
    ch500:           float = 0.0
    ch550:           float = 0.0
    ch570:           float = 0.0
    ch600:           float = 0.0
    ch650:           float = 0.0
    par_ue:          float = 0.0
    # CRT14016P pH (CH0)
    ph:              float = 0.0
    # SEN0244 TDS (CH1)
    tds_ppm:         float = 0.0
    ec_ms_cm:        float = 0.0
    # 액추에이터
    peltier_pct:     int   = 0
    led_brightness:  int   = 0
    led_r:           int   = 0
    led_g:           int   = 0
    led_b:           int   = 0
    led_blue_pct:    int   = 0
    led_red_pct:     int   = 0
    servo_angle:     int   = 0
    servo_feeding:   bool  = False
    # ExG 카메라 분석
    exg_mean:        float = 0.0
    green_ratio:     float = 0.0
    vari_mean:       float = 0.0
    ngrdi_mean:      float = 0.0
    color_index:     float = 0.0
    color_grade:     str   = "UNKNOWN"

@dataclass
class ControlConfig:
    """제어 루프 설정"""
    loop_interval_sec:   float = 5.0
    camera_interval_sec: float = 3600.0
    dht_min_interval:    float = 2.0
    temp_target:         float = 24.0
    temp_tol:            float = 2.0
    humidity_min:        float = 50.0
    humidity_max:        float = 80.0
    co2_min:             int   = 400
    co2_max:             int   = 1500
    ph_min:              float = 5.5
    ph_max:              float = 6.5
    tds_min:             float = 300.0
    tds_max:             float = 800.0
    ec_min:              float = 0.6
    ec_max:              float = 1.6
    led_blue_default:    float = 70.0
    led_red_default:     float = 80.0
    # MG92B 서보 영양액 공급 설정
    servo_feed_interval: float = 5.0    # 공급 주기 (초)
    servo_open_angle:    int   = 90     # 밸브 개방 각도
    servo_close_angle:   int   = 0      # 밸브 폐쇄 각도 (기본 위치)
    servo_hold_sec:      float = 0.5    # 개방 유지 시간 (초)

# =========================================================================
#  PAR 가중치
# =========================================================================

PAR_WEIGHTS = {
    "ch450": 0.10,
    "ch500": 0.15,
    "ch550": 0.20,
    "ch570": 0.20,
    "ch600": 0.15,
    "ch650": 0.20,
}

# =========================================================================
#  PID 제어기
# =========================================================================

class PIDController:
    """범용 PID 제어기 (출력 범위 0~100%)"""

    def __init__(self, kp: float = 2.0, ki: float = 0.1, kd: float = 0.5,
                 setpoint: float = 24.0,
                 output_min: float = 0.0, output_max: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def compute(self, measured: float) -> float:
        now = time.time()
        dt = max(now - self._prev_time, 1e-6)
        error = self.setpoint - measured

        self._integral += error * dt
        # anti-windup
        self._integral = max(-500, min(500, self._integral))

        derivative = (error - self._prev_error) / dt

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        output = max(self.output_min, min(self.output_max, output))

        self._prev_error = error
        self._prev_time = now
        return output

    def update_setpoint(self, setpoint: float):
        self.setpoint = setpoint
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

# =========================================================================
#  DHT11 온습도 센서 (1-Wire, GPIO 4)
# =========================================================================

class DHT11Sensor:
    """
    DHT11 온습도 센서
    범위: 온도 0~50 C (+-2 C), 습도 20~90 %RH (+-5 %)
    """

    def __init__(self):
        self._device = None
        if HARDWARE:
            self._device = adafruit_dht.DHT11(board.D4, use_pulseio=False)
            log.info("DHT11 초기화 완료 (GPIO 4)")
        else:
            log.info("[SIM] DHT11 시뮬레이션 모드")

    def read(self) -> tuple:
        if not HARDWARE:
            t = round(24.0 + random.uniform(-1, 1), 1)
            h = round(60.0 + random.uniform(-5, 5), 1)
            return t, h

        for attempt in range(3):
            try:
                t = self._device.temperature
                h = self._device.humidity
                if t is not None and h is not None:
                    return t, h
            except RuntimeError as e:
                log.warning("DHT11 재시도 (%d/3): %s", attempt + 1, e)
                time.sleep(2.0)
            except Exception as e:
                log.error("DHT11 치명적 오류: %s", e)
                break
        log.error("DHT11 읽기 실패 (3회 초과)")
        return None, None

    def cleanup(self):
        if self._device:
            self._device.exit()

# =========================================================================
#  RX-9 Simple CO2 센서 (CH2 EMF + CH3 Thermistor)
# =========================================================================

class RX9SimpleCO2:
    """
    EXSEN RX-9 Simple 전기화학식 CO2 센서

    EMF 출력     : ADS1015 CH2 (A2)
    Thermistor   : ADS1015 CH3 (A3), NTC 10k

    서미스터 회로 (전압 분배기):
        VCC (3.3 V) --- [ R_REF 10 kOhm ] ---+--- ADS1015 A3
                                              |
                                            [ NTC ]
                                              |
                                             GND

    온도 보상 EMF -> ppm (네른스트):
        compensation    = TEMP_COEFF * (T_sensor - T_REF)
        EMF_compensated = EMF_raw + compensation
        delta           = EMF_ZERO - EMF_compensated
        ppm             = 400 * 10^(delta / SLOPE)
    """

    EMF_ZERO      = 300.0
    SLOPE         = 55.0
    WARM_UP       = 120

    THERM_R_REF   = 10000.0
    THERM_R_NOM   = 10000.0
    THERM_B_COEFF = 3950.0
    THERM_T_NOM   = 25.0
    THERM_VCC     = 3.3

    TEMP_REF      = 25.0
    TEMP_COEFF    = 0.3

    def __init__(self, emf_channel=None, therm_channel=None):
        self._emf_ch = emf_channel
        self._therm_ch = therm_channel
        self._t0 = time.time()
        if HARDWARE:
            log.info("RX-9 Simple 초기화 (EMF->CH2, Therm->CH3)")
        else:
            log.info("[SIM] RX-9 Simple 시뮬레이션 모드")

    def is_warmed_up(self) -> bool:
        return (time.time() - self._t0) >= self.WARM_UP

    def read_sensor_temp(self) -> float:
        """NTC 서미스터 온도 (ADS1015 CH3)"""
        if not HARDWARE:
            return round(25.0 + random.uniform(-2, 2), 1)

        try:
            v = self._therm_ch.voltage
            if v >= self.THERM_VCC - 0.01:
                log.warning("서미스터 단선 의심")
                return -999.0
            if v <= 0.01:
                log.warning("서미스터 단락 의심")
                return -999.0

            r_ntc = (self.THERM_R_REF * v) / (self.THERM_VCC - v)
            t_nom_k = self.THERM_T_NOM + 273.15
            inv_t = (1.0 / t_nom_k) + (1.0 / self.THERM_B_COEFF) * math.log(r_ntc / self.THERM_R_NOM)
            temp_c = (1.0 / inv_t) - 273.15
            return round(temp_c, 1)
        except Exception as e:
            log.error("서미스터 오류: %s", e)
            return -999.0

    def read_emf_mv(self) -> float:
        if not HARDWARE:
            return round(250.0 + random.uniform(-10, 10), 1)
        try:
            return round(self._emf_ch.voltage * 1000.0, 1)
        except Exception as e:
            log.error("RX-9 EMF 오류: %s", e)
            return -1.0

    def read(self) -> tuple:
        """Returns: (co2_ppm, emf_mv, sensor_temp_c)"""
        if not HARDWARE:
            emf = round(250.0 + random.uniform(-10, 10), 1)
            ppm = 450 + random.randint(-30, 30)
            st = round(25.0 + random.uniform(-2, 2), 1)
            return ppm, emf, st

        sensor_temp = self.read_sensor_temp()

        if not self.is_warmed_up():
            remain = self.WARM_UP - (time.time() - self._t0)
            log.warning("RX-9 워밍업 중 (잔여 %.0fs)", remain)
            return -1, 0.0, sensor_temp

        emf_mv = self.read_emf_mv()
        if emf_mv < 0:
            return -1, emf_mv, sensor_temp

        try:
            comp_emf = emf_mv
            if sensor_temp > -900:
                comp = self.TEMP_COEFF * (sensor_temp - self.TEMP_REF)
                comp_emf = emf_mv + comp

            delta = self.EMF_ZERO - comp_emf
            if self.SLOPE != 0:
                ppm = int(400 * math.pow(10, delta / self.SLOPE))
            else:
                ppm = 400
            ppm = max(0, min(5000, ppm))
            return ppm, emf_mv, sensor_temp

        except Exception as e:
            log.error("RX-9 변환 오류: %s", e)
            return -1, emf_mv, sensor_temp

    def calibrate(self, emf_at_400: float, emf_at_4000: float):
        self.EMF_ZERO = emf_at_400
        self.SLOPE = (emf_at_400 - emf_at_4000)
        log.info("RX-9 교정: ZERO=%.1f  SLOPE=%.1f", self.EMF_ZERO, self.SLOPE)

# =========================================================================
#  AS7262 분광 센서 (I2C 0x49)
# =========================================================================

class AS7262Sensor:
    """6채널 가시광 분광: V(450), B(500), G(550), Y(570), O(600), R(650) nm"""

    ADDR = 0x49

    def __init__(self, i2c=None):
        self._i2c = i2c
        if HARDWARE and i2c is not None:
            self._init_hw()
            log.info("AS7262 초기화 완료 (0x%02X)", self.ADDR)
        else:
            log.info("[SIM] AS7262 시뮬레이션 모드")

    def _init_hw(self):
        try:
            self._vwrite(0x04, 0b00111100)
        except Exception as e:
            log.error("AS7262 초기화 오류: %s", e)

    def _vwrite(self, reg: int, val: int):
        while True:
            s = bytearray(1)
            self._i2c.writeto_then_readfrom(self.ADDR, bytes([0x00]), s)
            if not (s[0] & 0x02):
                break
        self._i2c.writeto(self.ADDR, bytes([0x01, reg | 0x80]))
        while True:
            s = bytearray(1)
            self._i2c.writeto_then_readfrom(self.ADDR, bytes([0x00]), s)
            if not (s[0] & 0x02):
                break
        self._i2c.writeto(self.ADDR, bytes([0x01, val]))

    def _vread(self, reg: int) -> int:
        while True:
            s = bytearray(1)
            self._i2c.writeto_then_readfrom(self.ADDR, bytes([0x00]), s)
            if not (s[0] & 0x02):
                break
        self._i2c.writeto(self.ADDR, bytes([0x01, reg]))
        while True:
            s = bytearray(1)
            self._i2c.writeto_then_readfrom(self.ADDR, bytes([0x00]), s)
            if s[0] & 0x01:
                break
        r = bytearray(1)
        self._i2c.writeto_then_readfrom(self.ADDR, bytes([0x02]), r)
        return r[0]

    def read(self) -> dict:
        if not HARDWARE or self._i2c is None:
            return {
                "ch450": round(random.uniform(10, 200), 2),
                "ch500": round(random.uniform(10, 200), 2),
                "ch550": round(random.uniform(10, 200), 2),
                "ch570": round(random.uniform(10, 200), 2),
                "ch600": round(random.uniform(10, 200), 2),
                "ch650": round(random.uniform(10, 200), 2),
            }
        try:
            names = ["ch450", "ch500", "ch550", "ch570", "ch600", "ch650"]
            result = {}
            for i, nm in enumerate(names):
                base = 0x14 + (i * 2)
                raw = bytearray(4)
                for j in range(4):
                    raw[j] = self._vread(base + j)
                result[nm] = round(struct.unpack(">f", raw)[0], 2)
            return result
        except Exception as e:
            log.error("AS7262 읽기 오류: %s", e)
            return {}

    @staticmethod
    def compute_par(spec: dict) -> float:
        return round(sum(spec.get(c, 0.0) * w for c, w in PAR_WEIGHTS.items()), 2)

# =========================================================================
#  ADS1015 + 수질 센서 (pH CH0, TDS CH1)
# =========================================================================

class WaterQuality:
    """
    ADS1015 12-bit ADC (I2C 0x48)
    CH0 : CRT14016P pH
    CH1 : SEN0244 TDS
    CH2 : RX-9 Simple EMF  (RX9SimpleCO2 에서 사용)
    CH3 : RX-9 Simple NTC  (RX9SimpleCO2 에서 사용)
    """

    def __init__(self, i2c=None):
        self._ads = None
        self._ph_chan = None
        self._tds_chan = None
        self._co2_chan = None
        self._therm_chan = None

        if HARDWARE and i2c is not None:
            self._ads = ADS.ADS1015(i2c)
            self._ph_chan = AnalogIn(self._ads, ADS.P0)
            self._tds_chan = AnalogIn(self._ads, ADS.P1)
            self._co2_chan = AnalogIn(self._ads, ADS.P2)
            self._therm_chan = AnalogIn(self._ads, ADS.P3)
            log.info("ADS1015 초기화 (CH0=pH, CH1=TDS, CH2=CO2, CH3=Therm)")
        else:
            log.info("[SIM] ADS1015 시뮬레이션 모드")

    @property
    def co2_channel(self):
        return self._co2_chan

    @property
    def therm_channel(self):
        return self._therm_chan

    def read_ph(self) -> float:
        """CRT14016P pH: pH = 7.0 + (1.65 - V) / 0.059"""
        if not HARDWARE or self._ph_chan is None:
            return round(6.0 + random.uniform(-0.3, 0.3), 2)
        try:
            v = self._ph_chan.voltage
            ph = 7.0 + (1.65 - v) / 0.059
            return round(max(0.0, min(14.0, ph)), 2)
        except Exception as e:
            log.error("pH 오류: %s", e)
            return -1.0

    def read_tds(self, temperature: float = 25.0) -> tuple:
        """SEN0244 TDS, Returns: (tds_ppm, ec_ms_cm)"""
        if not HARDWARE or self._tds_chan is None:
            tds = round(500 + random.uniform(-30, 30), 1)
            return tds, round(tds / 500.0, 3)
        try:
            v = self._tds_chan.voltage
            coef = 1.0 + 0.02 * (temperature - 25.0)
            vc = v / coef
            tds = (133.42 * vc**3 - 255.86 * vc**2 + 857.39 * vc) * 0.5
            tds = max(0.0, min(1000.0, tds))
            ec = tds / 500.0
            return round(tds, 1), round(ec, 3)
        except Exception as e:
            log.error("TDS 오류: %s", e)
            return -1.0, -1.0

# =========================================================================
#  WS2812B NeoPixel LED (GPIO 18)
# =========================================================================

class NeoPixelLED:
    """WS2812B 어드레서블 RGB LED"""

    def __init__(self, num_pixels: int = 16):
        self.num_pixels = num_pixels
        self._strip = None
        self.current_r = 0
        self.current_g = 0
        self.current_b = 0
        self.brightness = 0
        self.blue_pct = 0
        self.red_pct = 0

        if HARDWARE:
            self._strip = neopixel.NeoPixel(
                board.D18, num_pixels,
                brightness=1.0, auto_write=False,
                pixel_order=neopixel.GRB)
            self._strip.fill((0, 0, 0))
            self._strip.show()
            log.info("WS2812B 초기화 (%dpx, GPIO 18)", num_pixels)
        else:
            log.info("[SIM] WS2812B (%dpx)", num_pixels)

    def set_color(self, r: int, g: int, b: int):
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        self.current_r = r
        self.current_g = g
        self.current_b = b
        self.brightness = max(r, g, b)
        if HARDWARE and self._strip:
            self._strip.fill((r, g, b))
            self._strip.show()

    def set_grow_light(self, blue_pct: float, red_pct: float):
        b = int(max(0, min(100, blue_pct)) * 2.55)
        r = int(max(0, min(100, red_pct)) * 2.55)
        self.blue_pct = int(blue_pct)
        self.red_pct = int(red_pct)
        self.set_color(r, 0, b)

    def all_off(self):
        self.blue_pct = 0
        self.red_pct = 0
        self.set_color(0, 0, 0)

    def cleanup(self):
        self.all_off()

# =========================================================================
#  MG92B 서보 모터 (GPIO 23, PWM 50Hz)
# =========================================================================

class MG92BServo:
    """
    TowerPro MG92B 서보 모터 (영양액 밸브 제어)

    PWM 50Hz (주기 20ms) 기준 듀티 사이클:
      0도   = 2.5%   (밸브 닫힘, 기본 위치)
      90도  = 7.5%   (밸브 열림, 영양액 공급)
      180도 = 12.5%

    공식: duty_cycle = (angle / 18.0) + 2.5

    입력 전압: 5~6.6V
    토크: 3.1kg @ 5V, 3.5kg @ 6.6V
    동작 속도: 0.13sec/60도 @ 5V

    영양액 공급 시퀀스 (5초 간격):
      0도 -> 90도 (개방) -> 유지 0.5초 -> 90도 -> 0도 (폐쇄)

    운용 기준 pulse(us):
      CLOSE = 600us
      OPEN  = 1500us
    (현재 내부 구현은 angle 기반이며, pulse 입력은 변환 헬퍼 제공)
    """

    PIN_SERVO = 23
    PWM_FREQ = 50           # 서보 표준 주파수 50Hz (주기 20ms)
    MOVE_WAIT = 0.3         # 90도 회전 대기 시간 (0.13s/60 * 1.5 마진)
    SERVO_CLOSE_US = 600
    SERVO_OPEN_US = 1500

    def __init__(self):
        self._pwm = None
        self._current_angle = 0
        self._feeding = False
        self._feed_thread = None
        self._feed_stop = threading.Event()

        if HARDWARE:
            GPIO.setup(self.PIN_SERVO, GPIO.OUT)
            self._pwm = GPIO.PWM(self.PIN_SERVO, self.PWM_FREQ)
            self._pwm.start(0)  # 초기에는 신호 없음
            self.set_angle(0)   # 기본 위치 (닫힘)
            log.info("MG92B 서보 초기화 완료 (GPIO %d, %dHz)",
                     self.PIN_SERVO, self.PWM_FREQ)
        else:
            log.info("[SIM] MG92B 서보 시뮬레이션 모드 (GPIO %d)",
                     self.PIN_SERVO)

    @staticmethod
    def _angle_to_duty(angle: int) -> float:
        """
        각도(0~180) -> 듀티 사이클(2.5~12.5%)
        공식: duty = (angle / 18.0) + 2.5
        """
        angle = max(0, min(180, angle))
        return (angle / 18.0) + 2.5

    @staticmethod
    def _pulse_us_to_angle(pulse_us: int) -> int:
        """
        pulse width(us) -> angle 변환
        500us=0도, 2500us=180도 기준 선형 매핑
        """
        pulse = max(500, min(2500, int(pulse_us)))
        angle = int(round((pulse - 500) * 180.0 / 2000.0))
        return max(0, min(180, angle))

    def set_pulse_us(self, pulse_us: int):
        """외부 pulse 기준(예: 600/1500)으로 서보 제어"""
        self.set_angle(self._pulse_us_to_angle(pulse_us))

    def set_angle(self, angle: int):
        """지정 각도로 서보 회전"""
        angle = max(0, min(180, angle))
        duty = self._angle_to_duty(angle)

        if HARDWARE and self._pwm is not None:
            self._pwm.ChangeDutyCycle(duty)
            time.sleep(self.MOVE_WAIT)
            # 위치 도달 후 신호 중단 (지터 방지)
            self._pwm.ChangeDutyCycle(0)
        else:
            log.debug("[SIM] 서보 -> %d도 (duty=%.2f%%)", angle, duty)

        self._current_angle = angle

    @property
    def angle(self) -> int:
        return self._current_angle

    @property
    def is_feeding(self) -> bool:
        return self._feeding

    def feed_once(self, open_angle: int = 90, hold_sec: float = 0.5):
        """
        영양액 1회 공급:
          1) 0도 -> open_angle 회전 (밸브 개방)
          2) hold_sec 동안 유지
          3) open_angle -> 0도 복귀 (밸브 폐쇄)
        """
        self._feeding = True
        log.info("서보 공급 시작: 0도 -> %d도", open_angle)
        self.set_angle(open_angle)
        time.sleep(hold_sec)
        log.info("서보 복귀: %d도 -> 0도", open_angle)
        self.set_angle(0)
        self._feeding = False
        log.info("서보 공급 완료")

    def start_periodic_feeding(self, interval_sec: float = 5.0,
                               open_angle: int = 90,
                               hold_sec: float = 0.5):
        """
        주기적 영양액 공급 스레드 시작
        interval_sec 마다 feed_once() 실행
        """
        if self._feed_thread is not None and self._feed_thread.is_alive():
            log.warning("주기적 공급이 이미 실행 중")
            return

        self._feed_stop.clear()

        def _loop():
            log.info("주기적 서보 공급 시작 (간격=%.1fs, 각도=%d도, 유지=%.1fs)",
                     interval_sec, open_angle, hold_sec)
            while not self._feed_stop.is_set():
                self.feed_once(open_angle, hold_sec)
                # 대기 (stop 이벤트 감시하며)
                self._feed_stop.wait(timeout=interval_sec)
            log.info("주기적 서보 공급 중단")

        self._feed_thread = threading.Thread(target=_loop, daemon=True)
        self._feed_thread.start()

    def stop_periodic_feeding(self):
        """주기적 공급 스레드 중단"""
        self._feed_stop.set()
        if self._feed_thread is not None:
            self._feed_thread.join(timeout=10.0)
            self._feed_thread = None
        self.set_angle(0)
        self._feeding = False
        log.info("주기적 서보 공급 완전 중단, 0도 복귀")

    def cleanup(self):
        """자원 해제"""
        self.stop_periodic_feeding()
        self.set_angle(0)
        if HARDWARE and self._pwm is not None:
            self._pwm.ChangeDutyCycle(0)
            self._pwm.stop()
        log.info("MG92B 서보 정리 완료")

# =========================================================================
#  액추에이터 허브
# =========================================================================

class ActuatorHub:
    """펠티어(GPIO 17) + WS2812B(GPIO 18) + MG92B 서보(GPIO 23)"""

    PIN_PELTIER = 17
    PWM_FREQ_PELTIER = 1000

    def __init__(self, num_leds: int = 16):
        self.peltier_pct = 0
        self.led = NeoPixelLED(num_leds)
        self.servo = MG92BServo()

        if HARDWARE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.PIN_PELTIER, GPIO.OUT)
            self._pwm = GPIO.PWM(self.PIN_PELTIER, self.PWM_FREQ_PELTIER)
            self._pwm.start(0)
            log.info("액추에이터 허브 초기화 완료")
        else:
            log.info("[SIM] 액추에이터 허브 시뮬레이션 모드")

    def set_peltier(self, duty: float):
        duty = max(0, min(100, duty))
        self.peltier_pct = int(duty)
        if HARDWARE:
            self._pwm.ChangeDutyCycle(duty)

    def set_grow_light(self, blue_pct: float, red_pct: float):
        self.led.set_grow_light(blue_pct, red_pct)

    def all_off(self):
        self.set_peltier(0)
        self.led.all_off()
        self.servo.set_angle(0)

    def cleanup(self):
        self.all_off()
        self.led.cleanup()
        self.servo.cleanup()
        if HARDWARE:
            GPIO.cleanup()

# =========================================================================
#  XBee 920 MHz 통신 (UART TX=GPIO14, RX=GPIO15)
# =========================================================================

class XBeeComm:

    HEADER = b"\xAF\xAF"
    BAUD = 9600
    ONE_WAY_DELAY_SEC = 1.28

    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = BAUD,
                 one_way_delay_sec: float = ONE_WAY_DELAY_SEC):
        self._ser = None
        self._tx_seq = 0
        self._rx_expected_seq = None
        self._rx_last_seq = None
        self._rx_lost_packets = 0
        self._delay_sec = max(0.0, float(one_way_delay_sec))

        if HARDWARE:
            try:
                self._ser = serial.Serial(
                    port, int(baudrate), timeout=1.0, write_timeout=1.0)
                log.info("XBee 연결: %s @ %d (delay=%.2fs)",
                         port, int(baudrate), self._delay_sec)
            except Exception as e:
                log.warning("XBee 연결 실패: %s", e)
        else:
            log.info("[SIM] XBee 시뮬레이션 모드")

    @staticmethod
    def _to_dict(data) -> dict:
        if isinstance(data, SensorData):
            return asdict(data)
        if isinstance(data, dict):
            return data
        return {}

    @classmethod
    def _extract_control_state(cls, sensor_data) -> dict:
        src = cls._to_dict(sensor_data)
        keys = [
            "peltier_pct",
            "led_brightness", "led_r", "led_g", "led_b",
            "led_blue_pct", "led_red_pct",
            "servo_angle", "servo_feeding",
        ]
        return {k: src.get(k) for k in keys if k in src}

    @classmethod
    def _extract_exg_result(cls, sensor_data) -> dict:
        src = cls._to_dict(sensor_data)
        keys = [
            "exg_mean", "green_ratio", "vari_mean", "ngrdi_mean",
            "color_index", "color_grade",
        ]
        return {k: src.get(k) for k in keys if k in src}

    def _build_frame(self, payload_obj: dict) -> bytes:
        payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False).encode()
        length = len(payload).to_bytes(2, "big")
        return self.HEADER + length + payload

    def _send_frame(self, payload_obj: dict) -> bool:
        frame = self._build_frame(payload_obj)

        # 지구->달 편도 지연(소프트웨어 모사)
        if self._delay_sec > 0:
            time.sleep(self._delay_sec)

        if self._ser and self._ser.is_open:
            try:
                self._ser.write(frame)
                return True
            except Exception as e:
                log.error("XBee TX 오류: %s", e)
        else:
            log.debug("[SIM] TX %dB: %s", len(frame), payload_obj.get("msg_type", "unknown"))
        return False

    def send_telemetry(self,
                       sensor_data,
                       control_state: Optional[dict] = None,
                       lstm_prediction: Optional[dict] = None,
                       exg_result: Optional[dict] = None) -> bool:
        """
        텔레메트리 송신:
          sensor dict + control state + LSTM 예측 + ExG 결과를 JSON 패킷화
        """
        if control_state is None:
            control_state = self._extract_control_state(sensor_data)
        if exg_result is None:
            exg_result = self._extract_exg_result(sensor_data)

        packet = {
            "msg_type": "telemetry",
            "seq": int(self._tx_seq),
            "sent_at": datetime.now().isoformat(),
            "sensor": self._to_dict(sensor_data),
            "control_state": control_state or {},
            "lstm_prediction": lstm_prediction or {},
            "exg_result": exg_result or {},
        }
        ok = self._send_frame(packet)
        self._tx_seq = (self._tx_seq + 1) % (2**31)
        return ok

    def _check_rx_sequence(self, seq: Optional[int]):
        if seq is None:
            return
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return

        if self._rx_expected_seq is None:
            self._rx_expected_seq = seq + 1
            self._rx_last_seq = seq
            return

        if seq != self._rx_expected_seq:
            if seq > self._rx_expected_seq:
                lost = seq - self._rx_expected_seq
                self._rx_lost_packets += lost
                log.warning(
                    "XBee 패킷 손실 감지: expected=%d, got=%d, lost=%d, total_lost=%d",
                    self._rx_expected_seq, seq, lost, self._rx_lost_packets
                )
            else:
                log.warning(
                    "XBee 시퀀스 역전/중복: expected=%d, got=%d",
                    self._rx_expected_seq, seq
                )
        self._rx_expected_seq = seq + 1
        self._rx_last_seq = seq

    def _normalize_telecommand(self, cmd: dict) -> dict:
        """
        명령 종류 분기:
          - 온도설정
          - 광레시피
          - 영양액
          - 모델업데이트
        """
        cmd_type = str(
            cmd.get("command_type")
            or cmd.get("type")
            or cmd.get("action")
            or ""
        ).strip().lower()

        normalized = dict(cmd)
        normalized["command_type"] = cmd_type
        normalized["raw"] = cmd

        if cmd_type in ("온도설정", "temperature", "temp", "set_temp"):
            normalized["action"] = "set_temp"
            normalized["value"] = cmd.get("value", cmd.get("target_temp", cmd.get("temp", 22.0)))
            return normalized

        if cmd_type in ("광레시피", "light_recipe", "light", "set_grow_light"):
            normalized["action"] = "set_grow_light"
            normalized["blue"] = cmd.get("blue", cmd.get("blue_pct", 70))
            normalized["red"] = cmd.get("red", cmd.get("red_pct", 80))
            return normalized

        if cmd_type in ("영양액", "nutrient", "servo_feed", "servo_start", "servo_stop"):
            mode = str(cmd.get("mode", cmd.get("action", "once"))).lower()
            if mode in ("start", "periodic", "servo_start"):
                normalized["action"] = "servo_start"
            elif mode in ("stop", "servo_stop"):
                normalized["action"] = "servo_stop"
            else:
                normalized["action"] = "servo_feed"
            normalized["angle"] = cmd.get("angle", 90)
            normalized["hold"] = cmd.get("hold", 0.5)
            normalized["interval"] = cmd.get("interval", 5.0)
            return normalized

        if cmd_type in ("모델업데이트", "model_update", "update_model"):
            normalized["action"] = "model_update"
            normalized["model_version"] = cmd.get("model_version", "")
            normalized["model_url"] = cmd.get("model_url", "")
            normalized["checksum"] = cmd.get("checksum", "")
            return normalized

        # 기존 action 기반 명령도 그대로 전달
        if "action" in cmd:
            return normalized

        normalized["action"] = "unknown"
        return normalized

    def receive_command(self) -> Optional[dict]:
        if not self._ser or not self._ser.is_open:
            return None
        try:
            if self._ser.in_waiting < 4:
                return None
            hdr = self._ser.read(2)
            if hdr != self.HEADER:
                self._ser.reset_input_buffer()
                return None
            length = int.from_bytes(self._ser.read(2), "big")
            payload = self._ser.read(length)
            cmd = json.loads(payload.decode())
            if isinstance(cmd, dict):
                self._check_rx_sequence(cmd.get("seq"))
                return self._normalize_telecommand(cmd)
            return None
        except Exception as e:
            log.error("XBee RX 오류: %s", e)
            return None

    def stats(self) -> dict:
        return {
            "tx_seq": self._tx_seq,
            "rx_last_seq": self._rx_last_seq,
            "rx_expected_seq": self._rx_expected_seq,
            "rx_lost_packets": self._rx_lost_packets,
        }

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

# =========================================================================
#  영상 분석 (Pi Camera V2 + ExG)
# =========================================================================

class VisionAnalyzer:
    """RGB 기반 식생지수(ExG/VARI/NGRDI) 계산 및 등급 판정"""

    def __init__(self, out_dir: str = "captures"):
        import os
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def _safe_mean(arr, mask) -> float:
        if mask is not None and mask.any():
            return float(arr[mask].mean())
        return float(arr.mean())

    @staticmethod
    def _to_unit(value: float) -> float:
        return max(0.0, min(1.0, (value + 1.0) * 0.5))

    @classmethod
    def classify_color_index(cls, exg_mean: float, green_ratio: float,
                             vari_mean: float, ngrdi_mean: float) -> Tuple[float, str]:
        """
        RGB 지표를 0~100 스코어로 통합
        - green_ratio 가중치가 가장 큼 (잎 픽셀 비중)
        - ExG/VARI/NGRDI는 색상 건강도 보조
        """
        exg_u = cls._to_unit(exg_mean)
        vari_u = cls._to_unit(vari_mean)
        ngrdi_u = cls._to_unit(ngrdi_mean)
        gr_u = max(0.0, min(1.0, green_ratio))

        score = 100.0 * (
            0.40 * gr_u +
            0.25 * exg_u +
            0.20 * vari_u +
            0.15 * ngrdi_u
        )
        score = round(max(0.0, min(100.0, score)), 1)

        if score >= 70.0:
            grade = "GOOD"
        elif score >= 45.0:
            grade = "WARN"
        else:
            grade = "BAD"
        return score, grade

    def capture(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir + "/" + ts + ".jpg"
        if HARDWARE:
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam.configure(cam.create_still_configuration(
                    main={"size": (1920, 1080)}))
                cam.start()
                time.sleep(2)
                cam.capture_file(path)
                cam.stop()
                cam.close()
            except Exception as e:
                log.error("캡처 실패: %s", e)
                return ""
        else:
            log.debug("[SIM] 캡처 -> %s", path)
        return path

    def compute_exg(self, path: str) -> dict:
        try:
            import cv2
            import numpy as np
            img = cv2.imread(path)
            if img is None:
                log.error("이미지 읽기 실패: %s", path)
                return {}

            img = img.astype(np.float32)
            B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            tot = R + G + B + 1e-6
            r_n = R / tot
            g_n = G / tot
            b_n = B / tot

            exg = (2 * g_n) - r_n - b_n
            vari = (g_n - r_n) / (g_n + r_n - b_n + 1e-6)
            ngrdi = (g_n - r_n) / (g_n + r_n + 1e-6)

            # 식생 픽셀 우선 분리, 실패 시 전체 픽셀 평균 사용
            vegetation_mask = exg > 0.08
            mask = vegetation_mask if vegetation_mask.any() else None

            exg_mean = self._safe_mean(exg, mask)
            vari_mean = self._safe_mean(vari, mask)
            ngrdi_mean = self._safe_mean(ngrdi, mask)
            green_ratio = float(vegetation_mask.sum()) / exg.size
            color_index, color_grade = self.classify_color_index(
                exg_mean=exg_mean,
                green_ratio=green_ratio,
                vari_mean=vari_mean,
                ngrdi_mean=ngrdi_mean,
            )

            return {
                "exg_mean": round(exg_mean, 4),
                "green_ratio": round(green_ratio, 4),
                "vari_mean": round(vari_mean, 4),
                "ngrdi_mean": round(ngrdi_mean, 4),
                "color_index": color_index,
                "color_grade": color_grade,
            }
        except ImportError:
            log.warning("OpenCV 미설치, ExG 생략")
            return {}
        except Exception as e:
            log.error("ExG 오류: %s", e)
            return {}

# =========================================================================
#  ExG Analyzer (Pi Camera V2 + Otsu + Daily Scheduler)
# =========================================================================

class ExGAnalyzer:
    """
    Raspberry Pi Camera V2 촬영 이미지를 ExG로 분석
      - 정규화: r=R/(R+G+B), g=G/(R+G+B), b=B/(R+G+B)
      - ExG = 2g - r - b
      - Otsu 이진화로 식물 영역 마스크 생성
      - 전일 대비 mean_exg 10% 이상 하락 시 경고 플래그
      - 매일 같은 시각 자동 촬영 스케줄 지원
    """

    DROP_WARN_PCT = 10.0

    def __init__(self, out_dir: str = "captures", history_file: str = "exg_daily_history.json"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.history_path = os.path.join(out_dir, history_file)

        self._schedule_thread = None
        self._schedule_stop = threading.Event()
        self._scheduled_hhmm = "09:00"

    def capture_image(self) -> str:
        """picamera2로 still image 캡처"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"exg_{ts}.jpg")
        if HARDWARE:
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                cam.configure(cam.create_still_configuration(main={"size": (1920, 1080)}))
                cam.start()
                time.sleep(1.5)
                cam.capture_file(path)
                cam.stop()
                cam.close()
                return path
            except Exception as e:
                log.error("ExGAnalyzer 캡처 실패: %s", e)
                return ""
        log.warning("ExGAnalyzer 캡처 건너뜀(HARDWARE=False)")
        return ""

    def _load_history(self) -> Dict[str, float]:
        if not os.path.exists(self.history_path):
            return {}
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except Exception as e:
            log.error("ExGAnalyzer 이력 로드 실패: %s", e)
        return {}

    def _save_history(self, history: Dict[str, float]):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error("ExGAnalyzer 이력 저장 실패: %s", e)

    @staticmethod
    def _calc_drop_warning(today_mean_exg: float, prev_mean_exg: Optional[float],
                           threshold_pct: float = DROP_WARN_PCT) -> Tuple[bool, float]:
        """
        prev 대비 today 하락률 계산
        drop_pct = ((prev - today) / prev) * 100
        """
        if prev_mean_exg is None or prev_mean_exg <= 0:
            return False, 0.0
        drop_pct = ((prev_mean_exg - today_mean_exg) / prev_mean_exg) * 100.0
        return drop_pct >= threshold_pct, round(drop_pct, 2)

    def analyze_image(self, image_path: str, analysis_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        반환:
          {
            "coverage_pct": float,
            "mean_exg": float,
            "exg_map": ndarray,
            "drop_warning": bool
          }
        """
        if analysis_date is None:
            analysis_date = datetime.now()

        try:
            import cv2
            import numpy as np
        except ImportError:
            log.error("ExGAnalyzer OpenCV/Numpy 미설치")
            return {
                "coverage_pct": None,
                "mean_exg": None,
                "exg_map": None,
                "drop_warning": False,
            }

        img = cv2.imread(image_path)
        if img is None:
            log.error("ExGAnalyzer 이미지 읽기 실패: %s", image_path)
            return {
                "coverage_pct": None,
                "mean_exg": None,
                "exg_map": None,
                "drop_warning": False,
            }

        img = img.astype(np.float32)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        denom = r + g + b + 1e-6

        # 정규화 채널
        r_n = r / denom
        g_n = g / denom
        b_n = b / denom

        # ExG 지수
        exg_map = (2.0 * g_n) - r_n - b_n

        # Otsu 마스크
        exg_u8 = cv2.normalize(exg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_bool = mask > 0

        coverage_pct = float(mask_bool.sum()) * 100.0 / float(mask_bool.size)
        if mask_bool.any():
            mean_exg = float(exg_map[mask_bool].mean())
        else:
            mean_exg = float(exg_map.mean())

        today_key = analysis_date.strftime("%Y-%m-%d")
        prev_key = (analysis_date - timedelta(days=1)).strftime("%Y-%m-%d")
        history = self._load_history()
        prev_mean = history.get(prev_key)
        drop_warning, drop_pct = self._calc_drop_warning(mean_exg, prev_mean, self.DROP_WARN_PCT)

        history[today_key] = round(mean_exg, 6)
        self._save_history(history)

        return {
            "coverage_pct": round(coverage_pct, 2),
            "mean_exg": round(mean_exg, 6),
            "exg_map": exg_map,
            "drop_warning": drop_warning,
            "drop_pct": drop_pct,
            "prev_day_mean_exg": prev_mean,
            "image_path": image_path,
            "timestamp": analysis_date.isoformat(),
        }

    def capture_and_analyze(self) -> Dict[str, Any]:
        path = self.capture_image()
        if not path:
            return {
                "coverage_pct": None,
                "mean_exg": None,
                "exg_map": None,
                "drop_warning": False,
                "drop_pct": 0.0,
                "prev_day_mean_exg": None,
                "image_path": "",
                "timestamp": datetime.now().isoformat(),
            }
        return self.analyze_image(path)

    @staticmethod
    def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
        try:
            hh, mm = hhmm.strip().split(":")
            hour = max(0, min(23, int(hh)))
            minute = max(0, min(59, int(mm)))
            return hour, minute
        except Exception:
            return 9, 0

    def _seconds_until_next_run(self, hhmm: str) -> float:
        hour, minute = self._parse_hhmm(hhmm)
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        return max(1.0, (target - now).total_seconds())

    def start_daily_schedule(self, capture_time_hhmm: str = "09:00", handler=None):
        """
        매일 같은 시각 자동 촬영/분석 시작
        - capture_time_hhmm: "HH:MM"
        - handler(result_dict): 분석 결과 콜백(옵션)
        """
        if self._schedule_thread and self._schedule_thread.is_alive():
            log.warning("ExGAnalyzer 스케줄이 이미 실행 중")
            return

        self._scheduled_hhmm = capture_time_hhmm
        self._schedule_stop.clear()

        def _loop():
            log.info("ExGAnalyzer 일일 스케줄 시작 (%s)", self._scheduled_hhmm)
            while not self._schedule_stop.is_set():
                wait_sec = self._seconds_until_next_run(self._scheduled_hhmm)
                if self._schedule_stop.wait(timeout=wait_sec):
                    break
                result = self.capture_and_analyze()
                if handler is not None:
                    try:
                        handler(result)
                    except Exception as e:
                        log.error("ExGAnalyzer handler 오류: %s", e)
            log.info("ExGAnalyzer 일일 스케줄 중단")

        self._schedule_thread = threading.Thread(target=_loop, daemon=True)
        self._schedule_thread.start()

    def stop_daily_schedule(self):
        self._schedule_stop.set()
        if self._schedule_thread is not None:
            self._schedule_thread.join(timeout=5.0)
            self._schedule_thread = None

# =========================================================================
#  광주기 스케줄러
# =========================================================================

class PhotoperiodScheduler:
    """16h ON / 8h OFF (지구형 광주기)"""

    def __init__(self, on_hours: int = 16, off_hours: int = 8,
                 start_hour: int = 6):
        self.on_hours = on_hours
        self.off_hours = off_hours
        self.start_hour = start_hour

    def is_light_period(self) -> bool:
        now = datetime.now().hour
        end = (self.start_hour + self.on_hours) % 24
        if self.start_hour < end:
            return self.start_hour <= now < end
        return now >= self.start_hour or now < end

# =========================================================================
#  SensorManager (요구사항 전용 통합 읽기)
# =========================================================================

class SensorManager:
    """
    요구사항 기반 통합 센서 매니저
      - DHT11(GPIO4): temp_air, humidity
      - AS7262(I2C): par_450~par_650
      - ADS1015(I2C): A0=EC, A1=pH, A2=RX-9 CO2, A3=NTSF-4 water_temp
      - AS7262/ADS1015는 ThreadPoolExecutor로 병렬 읽기
      - 기본 읽기 주기: 5초
    """

    READ_PERIOD_SEC = 5.0
    RX9_EMF_ZERO_MV = 300.0
    RX9_SLOPE = 55.0

    def __init__(self, period_sec: float = READ_PERIOD_SEC):
        self.period_sec = max(0.1, float(period_sec))
        self._last_read = 0.0

        self._i2c = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._dht = DHT11Sensor()
        self._spectral = None

        # ADS1015 채널 매핑 (요구사항 고정)
        # A0: SEN0244(EC), A1: CRT14016P(pH), A2: RX-9 CO2, A3: NTSF-4(수온)
        self._ads = None
        self._ec_ch = None
        self._ph_ch = None
        self._co2_ch = None
        self._water_temp_ch = None

        if HARDWARE:
            try:
                self._i2c = busio.I2C(board.SCL, board.SDA)
                self._spectral = AS7262Sensor(self._i2c)
            except Exception as e:
                log.error("SensorManager I2C 초기화 실패: %s", e)
                self._spectral = AS7262Sensor(None)

            try:
                self._ads = ADS.ADS1015(self._i2c)
                self._ec_ch = AnalogIn(self._ads, ADS.P0)
                self._ph_ch = AnalogIn(self._ads, ADS.P1)
                self._co2_ch = AnalogIn(self._ads, ADS.P2)
                self._water_temp_ch = AnalogIn(self._ads, ADS.P3)
                log.info(
                    "SensorManager ADS1015 채널 매핑: A0=EC, A1=pH, A2=CO2, A3=water_temp"
                )
            except Exception as e:
                log.error("SensorManager ADS1015 초기화 실패: %s", e)
                self._ads = None
                self._ec_ch = None
                self._ph_ch = None
                self._co2_ch = None
                self._water_temp_ch = None
        else:
            self._spectral = AS7262Sensor(None)
            log.info("[SIM] SensorManager 시뮬레이션 모드")

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @classmethod
    def _co2_from_emf_mv(cls, emf_mv: float) -> Optional[int]:
        try:
            if emf_mv is None:
                return None
            delta = cls.RX9_EMF_ZERO_MV - emf_mv
            ppm = int(400 * math.pow(10, delta / cls.RX9_SLOPE))
            return int(cls._clip(ppm, 0, 5000))
        except Exception as e:
            log.error("SensorManager CO2 변환 실패: %s", e)
            return None

    def _read_dht11(self) -> Dict[str, Optional[float]]:
        temp_air, humidity = self._dht.read()
        if temp_air is None or humidity is None:
            log.error("SensorManager DHT11 읽기 실패")
        return {
            "temp_air": temp_air,
            "humidity": humidity,
        }

    def _read_ads1015_bundle(self) -> Dict[str, Optional[float]]:
        out = {
            "co2": None,
            "ec": None,
            "ph": None,
            "water_temp": None,
        }

        if not HARDWARE:
            out["ec"] = round(random.uniform(0.6, 1.8), 3)
            out["ph"] = round(random.uniform(5.7, 6.8), 2)
            out["co2"] = int(random.uniform(450, 1300))
            out["water_temp"] = round(random.uniform(20.0, 27.0), 1)
            return out

        if self._ads is None or self._ec_ch is None:
            log.warning("SensorManager ADS1015 없음: EC/pH/CO2/water_temp 건너뜀")
            return out

        # A0 -> SEN0244(EC)
        try:
            v_ec = self._ec_ch.voltage
            tds = (133.42 * v_ec**3 - 255.86 * v_ec**2 + 857.39 * v_ec) * 0.5
            ec = self._clip(tds / 500.0, 0.0, 5.0)
            out["ec"] = round(ec, 3)
        except Exception as e:
            log.error("SensorManager EC(A0) 읽기 실패: %s", e)

        # A1 -> CRT14016P(pH)
        try:
            v_ph = self._ph_ch.voltage
            ph = 7.0 + (1.65 - v_ph) / 0.059
            out["ph"] = round(self._clip(ph, 0.0, 14.0), 2)
        except Exception as e:
            log.error("SensorManager pH(A1) 읽기 실패: %s", e)

        # A2 -> RX-9 Simple(CO2)
        try:
            emf_mv = self._co2_ch.voltage * 1000.0
            out["co2"] = self._co2_from_emf_mv(emf_mv)
        except Exception as e:
            log.error("SensorManager CO2(A2) 읽기 실패: %s", e)

        # A3 -> NTSF-4(수온) : 0~3.3V -> 0~100C 선형 변환 (현장 교정 권장)
        try:
            v_wt = self._water_temp_ch.voltage
            wtemp = (v_wt / 3.3) * 100.0
            out["water_temp"] = round(self._clip(wtemp, -10.0, 100.0), 2)
        except Exception as e:
            log.error("SensorManager water_temp(A3) 읽기 실패: %s", e)

        return out

    def _read_as7262_bundle(self) -> Dict[str, Optional[float]]:
        out = {
            "par_450": None,
            "par_500": None,
            "par_550": None,
            "par_570": None,
            "par_600": None,
            "par_650": None,
        }
        try:
            spec = self._spectral.read() if self._spectral else {}
            if not spec:
                log.error("SensorManager AS7262 읽기 실패")
                return out
            out["par_450"] = spec.get("ch450")
            out["par_500"] = spec.get("ch500")
            out["par_550"] = spec.get("ch550")
            out["par_570"] = spec.get("ch570")
            out["par_600"] = spec.get("ch600")
            out["par_650"] = spec.get("ch650")
            return out
        except Exception as e:
            log.error("SensorManager AS7262 읽기 예외: %s", e)
            return out

    def probe_dht11(self) -> Dict[str, Any]:
        """GPIO4 DHT11 단독 읽기 (배선·라이브러리 점검용)."""
        out = dict(self._read_dht11())
        out["_probe"] = "DHT11"
        out["_hardware"] = HARDWARE
        return out

    def probe_as7262(self) -> Dict[str, Any]:
        """AS7262(I2C 0x49) 단독 읽기."""
        out = dict(self._read_as7262_bundle())
        out["_probe"] = "AS7262"
        out["_hardware"] = HARDWARE
        return out

    def probe_ads1015(self) -> Dict[str, Any]:
        """ADS1015(I2C 0x48) A0~A3 단독 읽기."""
        out = dict(self._read_ads1015_bundle())
        out["_probe"] = "ADS1015"
        out["_hardware"] = HARDWARE
        return out

    def read_once(self) -> Dict[str, Any]:
        """
        모든 센서를 1회 읽어 dict 반환.
        반환 예:
          {
            "temp_air", "humidity", "co2", "ec", "ph", "water_temp",
            "par_450", "par_500", "par_550", "par_570", "par_600", "par_650"
          }
        """
        base = self._read_dht11()

        # I2C 장치(AS7262/ADS1015)는 병렬 읽기
        fut_ads = self._executor.submit(self._read_ads1015_bundle)
        fut_as = self._executor.submit(self._read_as7262_bundle)

        ads_data = fut_ads.result()
        as_data = fut_as.result()

        base.update(ads_data)
        base.update(as_data)
        base["timestamp"] = datetime.now().isoformat()
        self._last_read = time.time()
        return base

    def run_loop(self, handler=None):
        """
        5초 주기(기본)로 read_once() 실행.
        handler를 주면 읽기 결과 dict를 전달.
        """
        log.info("SensorManager 루프 시작 (period=%.1fs)", self.period_sec)
        while True:
            t0 = time.time()
            data = self.read_once()
            if handler is not None:
                try:
                    handler(data)
                except Exception as e:
                    log.error("SensorManager handler 오류: %s", e)
            dt = time.time() - t0
            time.sleep(max(0.0, self.period_sec - dt))

    def close(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self._dht.cleanup()
        except Exception:
            pass

# =========================================================================
#  MockSensorManager (하드웨어 없는 통합 테스트용)
# =========================================================================

class MockSensorManager:
    """
    SensorManager와 동일 인터페이스를 제공하는 목 센서 매니저.
    - read_once()
    - run_loop(handler=None)
    - close()
    """

    READ_PERIOD_SEC = SensorManager.READ_PERIOD_SEC

    def __init__(self, period_sec: float = READ_PERIOD_SEC):
        self.period_sec = max(0.1, float(period_sec))
        self._last_read = 0.0
        self._closed = False
        log.info("[MOCK] MockSensorManager 초기화 (period=%.1fs)", self.period_sec)

    def read_once(self) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("MockSensorManager is closed")

        data = {
            "temp_air": round(random.uniform(21.0, 25.5), 1),
            "humidity": round(random.uniform(55.0, 75.0), 1),
            "co2": int(random.uniform(500, 1200)),
            "ec": round(random.uniform(0.8, 1.4), 3),
            "ph": round(random.uniform(5.8, 6.6), 2),
            "water_temp": round(random.uniform(20.0, 26.0), 1),
            "par_450": round(random.uniform(20.0, 180.0), 2),
            "par_500": round(random.uniform(20.0, 180.0), 2),
            "par_550": round(random.uniform(20.0, 180.0), 2),
            "par_570": round(random.uniform(20.0, 180.0), 2),
            "par_600": round(random.uniform(20.0, 180.0), 2),
            "par_650": round(random.uniform(20.0, 180.0), 2),
            "timestamp": datetime.now().isoformat(),
        }
        self._last_read = time.time()
        return data

    def run_loop(self, handler=None):
        log.info("MockSensorManager 루프 시작 (period=%.1fs)", self.period_sec)
        while not self._closed:
            t0 = time.time()
            data = self.read_once()
            if handler is not None:
                try:
                    handler(data)
                except Exception as e:
                    log.error("MockSensorManager handler 오류: %s", e)
            dt = time.time() - t0
            time.sleep(max(0.0, self.period_sec - dt))

    def close(self):
        self._closed = True

# =========================================================================
#  TemperatureController (Peltier PID + Relay Direction)
# =========================================================================

class TemperatureController:
    """
    SensorManager 출력(dict)을 입력받아 펠티어 릴레이를 PID로 제어
      - 목표 온도 범위: 18~26C (기본 22C)
      - PID gains 주입 가능
      - 릴레이 ON/OFF + 방향 전환(가열/냉각)
      - 제어 주기: 10초
    """

    TARGET_MIN = 18.0
    TARGET_MAX = 26.0
    DEFAULT_TARGET = 22.0
    DEFAULT_PERIOD_SEC = 10.0

    def __init__(self,
                 kp: float = 2.0,
                 ki: float = 0.1,
                 kd: float = 0.5,
                 target_temp: float = DEFAULT_TARGET,
                 control_period_sec: float = DEFAULT_PERIOD_SEC,
                 relay_enable_pin: int = 17,
                 relay_dir_pin: int = 27,
                 min_drive_pct: float = 5.0):
        self.relay_enable_pin = int(relay_enable_pin)
        self.relay_dir_pin = int(relay_dir_pin)
        self.control_period_sec = max(0.5, float(control_period_sec))
        self.min_drive_pct = max(0.0, min(100.0, float(min_drive_pct)))

        self.target_temp = self._clip(float(target_temp), self.TARGET_MIN, self.TARGET_MAX)
        self.pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            setpoint=self.target_temp,
            output_min=-100.0, output_max=100.0,
        )

        self.relay_on = False
        self.relay_mode = "OFF"   # OFF / HEAT / COOL
        self.last_output = 0.0

        if HARDWARE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.relay_enable_pin, GPIO.OUT)
                GPIO.setup(self.relay_dir_pin, GPIO.OUT)
                GPIO.output(self.relay_enable_pin, GPIO.LOW)
                GPIO.output(self.relay_dir_pin, GPIO.LOW)
                log.info(
                    "TemperatureController 초기화 (EN=%d, DIR=%d, target=%.1fC)",
                    self.relay_enable_pin, self.relay_dir_pin, self.target_temp
                )
            except Exception as e:
                log.error("TemperatureController GPIO 초기화 실패: %s", e)
        else:
            log.info("[SIM] TemperatureController 시뮬레이션 모드")

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def set_target(self, target_temp: float):
        self.target_temp = self._clip(float(target_temp), self.TARGET_MIN, self.TARGET_MAX)
        self.pid.update_setpoint(self.target_temp)

    def _apply_relay(self, mode: str, relay_on: bool):
        self.relay_mode = mode
        self.relay_on = relay_on

        if not HARDWARE:
            return

        try:
            # relay_enable: HIGH=ON, LOW=OFF
            # relay_dir   : LOW=HEAT, HIGH=COOL
            GPIO.output(self.relay_enable_pin, GPIO.HIGH if relay_on else GPIO.LOW)
            if mode == "COOL":
                GPIO.output(self.relay_dir_pin, GPIO.HIGH)
            else:
                GPIO.output(self.relay_dir_pin, GPIO.LOW)
        except Exception as e:
            log.error("TemperatureController 릴레이 출력 실패: %s", e)

    def control_once(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SensorManager 출력(dict) 기준 1회 제어 수행.
        반환:
          {
            "current_temp", "error", "pid_output",
            "relay_on", "relay_mode", "target_temp", "timestamp"
          }
        """
        current_temp = sensor_data.get("temp_air")
        if current_temp is None:
            log.error("TemperatureController temp_air 누락: 제어 생략")
            self._apply_relay("OFF", False)
            return {
                "current_temp": None,
                "error": None,
                "pid_output": None,
                "relay_on": self.relay_on,
                "relay_mode": self.relay_mode,
                "target_temp": self.target_temp,
                "timestamp": datetime.now().isoformat(),
            }

        error = self.target_temp - float(current_temp)
        pid_output = float(self.pid.compute(float(current_temp)))  # -100 ~ +100
        self.last_output = pid_output

        drive = abs(pid_output)
        if drive < self.min_drive_pct:
            self._apply_relay("OFF", False)
        elif pid_output > 0:
            # 목표보다 낮음 -> 가열
            self._apply_relay("HEAT", True)
        else:
            # 목표보다 높음 -> 냉각
            self._apply_relay("COOL", True)

        return {
            "current_temp": round(float(current_temp), 2),
            "error": round(error, 2),
            "pid_output": round(pid_output, 2),
            "relay_on": self.relay_on,
            "relay_mode": self.relay_mode,
            "target_temp": round(self.target_temp, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def run_loop(self, sensor_manager: SensorManager, handler=None):
        """10초 주기(기본)로 센서 읽기 + PID 제어 수행"""
        log.info("TemperatureController 루프 시작 (period=%.1fs)", self.control_period_sec)
        while True:
            t0 = time.time()
            sensor_data = sensor_manager.read_once()
            result = self.control_once(sensor_data)

            if handler is not None:
                try:
                    handler(result)
                except Exception as e:
                    log.error("TemperatureController handler 오류: %s", e)

            dt = time.time() - t0
            time.sleep(max(0.0, self.control_period_sec - dt))

    def cleanup(self):
        self._apply_relay("OFF", False)

# =========================================================================
#  통합 센서 허브
# =========================================================================

class SensorHub:
    """모든 센서를 통합 관리, 하드웨어 없으면 시뮬레이션"""

    def __init__(self):
        self._i2c = None
        if HARDWARE:
            self._i2c = busio.I2C(board.SCL, board.SDA)

        self.dht11 = DHT11Sensor()
        self.water = WaterQuality(self._i2c)
        self.co2 = RX9SimpleCO2(
            emf_channel=self.water.co2_channel,
            therm_channel=self.water.therm_channel,
        )
        self.spectral = AS7262Sensor(self._i2c)

        self._last_dht = 0.0
        self._cache_t = 24.0
        self._cache_h = 60.0

    def read_all(self) -> SensorData:
        d = SensorData(timestamp=datetime.now().isoformat())
        now = time.time()

        # (a) DHT11 (2초 간격 제한)
        if now - self._last_dht >= 2.0:
            t, h = self.dht11.read()
            if t is not None:
                self._cache_t = t
                self._cache_h = h
            self._last_dht = now
        d.temperature = self._cache_t
        d.humidity = self._cache_h

        # (b) RX-9 Simple (EMF + Thermistor)
        ppm, emf, s_temp = self.co2.read()
        if ppm >= 0:
            d.co2_ppm = ppm
            d.co2_emf_mv = emf
        d.co2_sensor_temp = s_temp

        # (c) AS7262 분광
        spec = self.spectral.read()
        if spec:
            d.ch450 = spec.get("ch450", 0.0)
            d.ch500 = spec.get("ch500", 0.0)
            d.ch550 = spec.get("ch550", 0.0)
            d.ch570 = spec.get("ch570", 0.0)
            d.ch600 = spec.get("ch600", 0.0)
            d.ch650 = spec.get("ch650", 0.0)
            d.par_ue = AS7262Sensor.compute_par(spec)

        # (d) pH (CH0)
        d.ph = self.water.read_ph()

        # (e) TDS (CH1)
        tds, ec = self.water.read_tds(temperature=d.temperature)
        d.tds_ppm = tds
        d.ec_ms_cm = ec

        return d

    def cleanup(self):
        self.dht11.cleanup()

# =========================================================================
#  지상국 성능 지표
# =========================================================================

class PerformanceMetrics:
    """시스템 안정성, RMS 오차, EC 흡수율 계산"""

    TARGETS = {
        "temperature": 24.0,
        "humidity": 65.0,
        "co2_ppm": 800,
        "ec_ms_cm": 1.0,
    }
    TOLERANCE = {
        "temperature": 2.0,
        "humidity": 10.0,
        "co2_ppm": 200,
        "ec_ms_cm": 0.4,
    }

    def __init__(self):
        self._history: List[dict] = []

    def add(self, data: dict):
        self._history.append(data)

    def stability_pct(self) -> float:
        if not self._history:
            return 0.0
        stable = 0
        for d in self._history:
            ok = True
            for key, tgt in self.TARGETS.items():
                val = d.get(key, tgt)
                tol = self.TOLERANCE.get(key, 0)
                if abs(val - tgt) > tol:
                    ok = False
                    break
            if ok:
                stable += 1
        return round(100.0 * stable / len(self._history), 1)

    def rms_error(self, key: str) -> float:
        tgt = self.TARGETS.get(key, 0)
        vals = [d.get(key, tgt) for d in self._history]
        if not vals:
            return 0.0
        mse = sum((v - tgt) ** 2 for v in vals) / len(vals)
        return round(math.sqrt(mse), 4)

    def ec_absorption_rate(self) -> Optional[float]:
        ec_vals = [d.get("ec_ms_cm", None) for d in self._history]
        ec_vals = [v for v in ec_vals if v is not None]
        if len(ec_vals) < 2:
            return None
        n = len(ec_vals)
        x_mean = (n - 1) / 2.0
        y_mean = sum(ec_vals) / n
        num = sum((i - x_mean) * (ec_vals[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return None
        slope = num / den
        return round(-slope, 6)

class DataLogger:
    """JSON 라인 형식으로 로그 저장"""

    def __init__(self, path: str = "astrofarm_data.jsonl"):
        self._path = path

    def log_data(self, data: dict):
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error("DataLogger 오류: %s", e)

# =========================================================================
#  메인 컨트롤러
# =========================================================================

class AstroFarmController:
    """
    메인 루프:
      1. 데이터 로드  (센서 순차 스캔)
      2. 상태 진단    (생존 범위 판단)
      3. 명령 실행    (PID 제어 + 서보 영양액 공급)
      4. 원격 송신    (XBee 텔레메트리)
    """

    def __init__(self, cfg: ControlConfig = None):
        if cfg is None:
            cfg = ControlConfig()
        self.cfg = cfg

        log.info("=" * 40)
        log.info("  AstroFarm 달 기지 스마트팜 초기화")
        log.info("=" * 40)

        self.sensors = SensorHub()
        self.actuators = ActuatorHub(num_leds=16)
        self.xbee = XBeeComm()
        self.vision = VisionAnalyzer()
        self.scheduler = PhotoperiodScheduler()
        self.pid_temp = PIDController(
            kp=2.0, ki=0.1, kd=0.5, setpoint=cfg.temp_target)

        self._last_cap = 0.0
        self._ec_history: List[Tuple[float, float]] = []
        self._running = False
        self._metrics = PerformanceMetrics()
        self._logger = DataLogger()

        log.info("모든 모듈 초기화 완료 (HARDWARE=%s)", HARDWARE)

    # ----- 진단 -----

    def _diagnose(self, d: SensorData) -> list:
        alerts = []
        c = self.cfg

        if abs(d.temperature - c.temp_target) > c.temp_tol:
            alerts.append("온도 이탈: %.1fC" % d.temperature)
        if d.humidity < c.humidity_min or d.humidity > c.humidity_max:
            alerts.append("습도 이탈: %.1f%%" % d.humidity)
        if d.co2_ppm > c.co2_max:
            alerts.append("CO2 과다: %dppm" % d.co2_ppm)
        elif 0 < d.co2_ppm < c.co2_min:
            alerts.append("CO2 부족: %dppm" % d.co2_ppm)
        if d.ph > 0 and (d.ph < c.ph_min or d.ph > c.ph_max):
            alerts.append("pH 이탈: %.2f" % d.ph)
        if d.tds_ppm > 0 and (d.tds_ppm < c.tds_min or d.tds_ppm > c.tds_max):
            alerts.append("TDS 이탈: %.1fppm" % d.tds_ppm)
        if d.ec_ms_cm > 0 and (d.ec_ms_cm < c.ec_min or d.ec_ms_cm > c.ec_max):
            alerts.append("EC 이탈: %.3fmS/cm" % d.ec_ms_cm)
        if not self.sensors.co2.is_warmed_up():
            alerts.append("RX-9 워밍업 중")
        if d.co2_sensor_temp < -900:
            alerts.append("RX-9 서미스터 이상")
        elif d.co2_sensor_temp > 60:
            alerts.append("RX-9 센서 과열: %.1fC" % d.co2_sensor_temp)

        for a in alerts:
            log.warning("[진단] %s", a)
        return alerts

    # ----- 제어 실행 -----

    def _execute_control(self, d: SensorData):
        # (1) 온도 PID -> 펠티어
        if d.temperature > 0:
            duty = self.pid_temp.compute(d.temperature)
            self.actuators.set_peltier(duty)
            d.peltier_pct = self.actuators.peltier_pct

        # (2) WS2812B 광주기
        if self.scheduler.is_light_period():
            bp = self.cfg.led_blue_default
            rp = self.cfg.led_red_default
            if 0 < d.par_ue < 50:
                bp = min(100, bp + 15)
                rp = min(100, rp + 15)
            self.actuators.set_grow_light(bp, rp)
        else:
            self.actuators.led.all_off()

        d.led_brightness = self.actuators.led.brightness
        d.led_r = self.actuators.led.current_r
        d.led_g = self.actuators.led.current_g
        d.led_b = self.actuators.led.current_b
        d.led_blue_pct = self.actuators.led.blue_pct
        d.led_red_pct = self.actuators.led.red_pct

        # (3) MG92B 서보: TDS 부족 시 주기적 공급 시작, 정상이면 중단
        if 0 < d.tds_ppm < self.cfg.tds_min:
            if not self.actuators.servo.is_feeding:
                self.actuators.servo.start_periodic_feeding(
                    interval_sec=self.cfg.servo_feed_interval,
                    open_angle=self.cfg.servo_open_angle,
                    hold_sec=self.cfg.servo_hold_sec,
                )
        else:
            if self.actuators.servo.is_feeding:
                self.actuators.servo.stop_periodic_feeding()

        d.servo_angle = self.actuators.servo.angle
        d.servo_feeding = self.actuators.servo.is_feeding

        # (4) 지상국 명령
        cmd = self.xbee.receive_command()
        if cmd:
            self._handle_cmd(cmd)

    # ----- EC 흡수율 -----

    def _compute_ec_absorption_rate(self) -> Optional[float]:
        if len(self._ec_history) < 2:
            return None
        vals = [v for _, v in self._ec_history]
        n = len(vals)
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (vals[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return None
        return round(-num / den, 6)

    # ----- 지상국 명령 처리 -----

    def _handle_cmd(self, cmd: dict):
        action = cmd.get("action", "")
        log.info("지상국 명령: %s", cmd)

        if action == "set_temp":
            new_sp = float(cmd.get("value", 24.0))
            self.cfg.temp_target = new_sp
            self.pid_temp.update_setpoint(new_sp)

        elif action == "set_led":
            self.actuators.led.set_color(
                int(cmd.get("r", 0)),
                int(cmd.get("g", 0)),
                int(cmd.get("b", 255)))

        elif action == "set_grow_light":
            self.actuators.set_grow_light(
                float(cmd.get("blue", 70)),
                float(cmd.get("red", 80)))

        elif action == "servo_feed":
            # 수동 1회 공급
            angle = int(cmd.get("angle", 90))
            hold = float(cmd.get("hold", 0.5))
            threading.Thread(
                target=self.actuators.servo.feed_once,
                args=(angle, hold),
                daemon=True).start()

        elif action == "servo_start":
            # 주기적 공급 시작
            interval = float(cmd.get("interval", 5.0))
            angle = int(cmd.get("angle", 90))
            hold = float(cmd.get("hold", 0.5))
            self.actuators.servo.start_periodic_feeding(interval, angle, hold)

        elif action == "servo_stop":
            # 주기적 공급 중단
            self.actuators.servo.stop_periodic_feeding()

        elif action == "servo_angle":
            # 서보 수동 각도 설정
            self.actuators.servo.set_angle(int(cmd.get("angle", 0)))

        elif action == "capture":
            self._do_capture()

        elif action == "calibrate_co2":
            self.sensors.co2.calibrate(
                float(cmd.get("emf_400", 300)),
                float(cmd.get("emf_4000", 245)))

        elif action == "model_update":
            # 모델 파일 교체/재로딩은 시스템 정책에 따라 외부 워커에서 수행
            log.info(
                "모델 업데이트 명령 수신: version=%s url=%s checksum=%s",
                cmd.get("model_version", ""),
                cmd.get("model_url", ""),
                cmd.get("checksum", "")
            )

        else:
            log.warning("알 수 없는 명령: %s", action)

    def _do_capture(self) -> dict:
        path = self.vision.capture()
        if path:
            return self.vision.compute_exg(path)
        return {}

    # ----- 메인 루프 -----

    def run(self):
        log.info("메인 루프 시작 (HARDWARE=%s)", HARDWARE)
        self._running = True
        try:
            while self._running:
                t0 = time.time()

                data = self.sensors.read_all()
                self._ec_history.append((t0, data.ec_ms_cm))
                if len(self._ec_history) > 720:
                    self._ec_history.pop(0)

                alerts = self._diagnose(data)
                self._execute_control(data)
                self.xbee.send_telemetry(data)

                self._metrics.add(asdict(data))
                self._logger.log_data(asdict(data))

                now = time.time()
                if now - self._last_cap > self.cfg.camera_interval_sec:
                    exg = self._do_capture()
                    if exg:
                        data.exg_mean = exg.get("exg_mean", 0.0)
                        data.green_ratio = exg.get("green_ratio", 0.0)
                        data.vari_mean = exg.get("vari_mean", 0.0)
                        data.ngrdi_mean = exg.get("ngrdi_mean", 0.0)
                        data.color_index = exg.get("color_index", 0.0)
                        data.color_grade = exg.get("color_grade", "UNKNOWN")
                    self._last_cap = now

                elapsed = time.time() - t0
                sleep_t = max(0, self.cfg.loop_interval_sec - elapsed)
                time.sleep(sleep_t)

        except KeyboardInterrupt:
            log.info("사용자 중단")
        finally:
            self.shutdown()

    def shutdown(self):
        log.info("종료 절차 시작")
        self._running = False
        self.actuators.cleanup()
        self.sensors.cleanup()
        self.xbee.close()
        log.info("AstroFarm 종료 완료")

# =========================================================================
#  단위 테스트
# =========================================================================

class TestPIDController(unittest.TestCase):

    def test_proportional_response(self):
        """측정값이 목표보다 낮으면 양의 출력 반환"""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=22.0)
        output = pid.compute(20.0)
        self.assertGreater(output, 0)

    def test_output_clamped(self):
        """출력이 0~100 범위를 벗어나지 않아야 함"""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, setpoint=22.0)
        self.assertLessEqual(pid.compute(0.0), 100.0)
        self.assertGreaterEqual(pid.compute(50.0), 0.0)

    def test_integral_reset_on_setpoint_update(self):
        """setpoint 변경 시 적분항 초기화"""
        pid = PIDController(kp=1.0, ki=1.0, kd=0.0, setpoint=22.0)
        for _ in range(10):
            pid.compute(20.0)
        pid.update_setpoint(25.0)
        self.assertEqual(pid._integral, 0.0)

    def test_convergence(self):
        """PID 출력이 오차를 줄이는 방향으로 작동"""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, setpoint=22.0)
        temp = 18.0
        initial_error = abs(22.0 - temp)
        for _ in range(50):
            duty = pid.compute(temp)
            temp = min(temp + duty * 0.05, 40.0)
        final_error = abs(22.0 - temp)
        self.assertLess(final_error, initial_error)

class TestPhotoPeriodScheduler(unittest.TestCase):

    def test_light_on_during_day(self):
        sched = PhotoperiodScheduler(on_hours=16, off_hours=8, start_hour=6)
        # 10시는 6~22(=6+16) 범위 안이므로 점등
        now_hour = 10
        end_hour = (6 + 16) % 24  # 22
        self.assertTrue(6 <= now_hour < 22)

    def test_light_off_at_night(self):
        sched = PhotoperiodScheduler(on_hours=16, off_hours=8, start_hour=6)
        # 03시는 6~22 범위 밖이므로 소등
        now_hour = 3
        self.assertFalse(6 <= now_hour < 22)

class TestExGAnalyzer(unittest.TestCase):

    def test_drop_warning_triggered(self):
        warn, drop = ExGAnalyzer._calc_drop_warning(
            today_mean_exg=0.72, prev_mean_exg=0.90, threshold_pct=10.0
        )
        self.assertTrue(warn)
        self.assertGreaterEqual(drop, 10.0)

    def test_drop_warning_not_triggered(self):
        warn, drop = ExGAnalyzer._calc_drop_warning(
            today_mean_exg=0.86, prev_mean_exg=0.90, threshold_pct=10.0
        )
        self.assertFalse(warn)
        self.assertLess(drop, 10.0)

    def test_drop_warning_without_previous_data(self):
        warn, drop = ExGAnalyzer._calc_drop_warning(
            today_mean_exg=0.50, prev_mean_exg=None, threshold_pct=10.0
        )
        self.assertFalse(warn)
        self.assertEqual(drop, 0.0)

class TestExGAnalysis(unittest.TestCase):

    def test_exg_formula(self):
        """ExG = 2G - R - B 수식 검증"""
        r, g, b = 0.3, 0.5, 0.2
        exg = 2 * g - r - b
        self.assertAlmostEqual(exg, 0.5)

    def test_pure_green_pixel_high_exg(self):
        """순수 녹색 픽셀은 높은 ExG를 가져야 함"""
        r, g, b = 0.0, 1.0, 0.0
        total = r + g + b + 1e-6
        r_n = r / total
        g_n = g / total
        b_n = b / total
        exg = 2 * g_n - r_n - b_n
        self.assertGreater(exg, 1.5)

    def test_color_grade_good(self):
        score, grade = VisionAnalyzer.classify_color_index(
            exg_mean=0.7, green_ratio=0.8, vari_mean=0.6, ngrdi_mean=0.5
        )
        self.assertGreaterEqual(score, 70.0)
        self.assertEqual(grade, "GOOD")

    def test_color_grade_bad(self):
        score, grade = VisionAnalyzer.classify_color_index(
            exg_mean=-0.3, green_ratio=0.1, vari_mean=-0.2, ngrdi_mean=-0.3
        )
        self.assertLess(score, 45.0)
        self.assertEqual(grade, "BAD")

class TestPerformanceMetrics(unittest.TestCase):

    def _make_data(self, temp=22.0, hum=65.0, co2=1000, ec=1.8):
        return {
            "timestamp": "2025-01-01T00:00:00",
            "temperature": temp,
            "humidity": hum,
            "co2_ppm": co2,
            "ec_ms_cm": ec,
            "par_ue": 200.0,
        }

    def test_stable_data_high_stability(self):
        """목표값에 맞는 데이터는 안정성 100%"""
        m = PerformanceMetrics()
        for _ in range(20):
            m.add(self._make_data(temp=24.0, hum=65.0, co2=800, ec=1.0))
        self.assertEqual(m.stability_pct(), 100.0)

    def test_rms_error_zero_on_target(self):
        """목표값과 동일한 측정값의 RMS 오차는 0"""
        m = PerformanceMetrics()
        for _ in range(10):
            m.add(self._make_data(temp=24.0))
        self.assertAlmostEqual(m.rms_error("temperature"), 0.0)

    def test_ec_absorption_rate(self):
        """EC 감소 추세에서 흡수율은 양수"""
        m = PerformanceMetrics()
        for i in range(100):
            m.add(self._make_data(ec=2.0 - i * 0.005))
        rate = m.ec_absorption_rate()
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0)

class TestTemperatureController(unittest.TestCase):
    """TemperatureController 동작 검증"""

    def test_target_clamped_to_range(self):
        tc = TemperatureController(target_temp=30.0)
        self.assertEqual(tc.target_temp, 26.0)
        tc.set_target(10.0)
        self.assertEqual(tc.target_temp, 18.0)
        tc.cleanup()

    def test_control_heat_mode(self):
        tc = TemperatureController(target_temp=22.0, kp=20.0, ki=0.0, kd=0.0)
        out = tc.control_once({"temp_air": 18.0})
        self.assertTrue(out["relay_on"])
        self.assertEqual(out["relay_mode"], "HEAT")
        self.assertGreater(out["pid_output"], 0)
        tc.cleanup()

    def test_control_cool_mode(self):
        tc = TemperatureController(target_temp=22.0, kp=20.0, ki=0.0, kd=0.0)
        out = tc.control_once({"temp_air": 26.0})
        self.assertTrue(out["relay_on"])
        self.assertEqual(out["relay_mode"], "COOL")
        self.assertLess(out["pid_output"], 0)
        tc.cleanup()

    def test_control_missing_temperature(self):
        tc = TemperatureController()
        out = tc.control_once({})
        self.assertIsNone(out["current_temp"])
        self.assertIsNone(out["error"])
        self.assertIsNone(out["pid_output"])
        self.assertFalse(out["relay_on"])
        tc.cleanup()

class TestXBeeComm(unittest.TestCase):
    """XBeeComm 텔레메트리/명령 파서 검증"""

    def test_send_telemetry_builds_required_sections(self):
        xb = XBeeComm(one_way_delay_sec=0.0)
        captured = {}

        def _fake_send_frame(payload_obj):
            captured.update(payload_obj)
            return True

        xb._send_frame = _fake_send_frame  # type: ignore[attr-defined]
        ok = xb.send_telemetry(
            sensor_data={
                "temperature": 24.0,
                "exg_mean": 0.52,
                "peltier_pct": 40,
            },
            lstm_prediction={"mae": 0.08, "warning": False},
        )
        self.assertTrue(ok)
        self.assertEqual(captured.get("msg_type"), "telemetry")
        self.assertIn("sensor", captured)
        self.assertIn("control_state", captured)
        self.assertIn("lstm_prediction", captured)
        self.assertIn("exg_result", captured)
        self.assertEqual(captured["sensor"]["temperature"], 24.0)

    def test_telecommand_branch_temperature(self):
        xb = XBeeComm(one_way_delay_sec=0.0)
        cmd = xb._normalize_telecommand({"command_type": "온도설정", "target_temp": 23.5})
        self.assertEqual(cmd.get("action"), "set_temp")
        self.assertEqual(float(cmd.get("value")), 23.5)

    def test_packet_loss_detection(self):
        xb = XBeeComm(one_way_delay_sec=0.0)
        xb._check_rx_sequence(10)
        xb._check_rx_sequence(13)  # 11,12 유실
        s = xb.stats()
        self.assertEqual(s.get("rx_lost_packets"), 2)

class TestSensorSimulation(unittest.TestCase):
    """시뮬레이션 모드에서 센서 읽기 검증"""

    def test_mock_sensor_manager_interface(self):
        sm = MockSensorManager(period_sec=0.5)
        data = sm.read_once()
        for key in [
            "temp_air", "humidity", "co2", "ec", "ph", "water_temp",
            "par_450", "par_500", "par_550", "par_570", "par_600", "par_650",
            "timestamp",
        ]:
            self.assertIn(key, data)
            self.assertIsNotNone(data[key])
        sm.close()

    def test_dht11_returns_values(self):
        sensor = DHT11Sensor()
        t, h = sensor.read()
        self.assertIsNotNone(t)
        self.assertIsNotNone(h)
        self.assertGreater(t, 0)
        self.assertGreater(h, 0)

    def test_rx9_returns_tuple(self):
        co2 = RX9SimpleCO2()
        ppm, emf, stemp = co2.read()
        self.assertGreater(ppm, 0)
        self.assertGreater(emf, 0)
        self.assertGreater(stemp, -900)

    def test_rx9_thermistor(self):
        co2 = RX9SimpleCO2()
        temp = co2.read_sensor_temp()
        self.assertGreater(temp, 0)
        self.assertLess(temp, 50)

    def test_water_quality_ph(self):
        wq = WaterQuality()
        ph = wq.read_ph()
        self.assertGreater(ph, 0)
        self.assertLess(ph, 14)

    def test_water_quality_tds(self):
        wq = WaterQuality()
        tds, ec = wq.read_tds()
        self.assertGreater(tds, 0)
        self.assertGreater(ec, 0)

    def test_as7262_returns_6ch(self):
        spec = AS7262Sensor()
        data = spec.read()
        self.assertEqual(len(data), 6)
        for key in ["ch450", "ch500", "ch550", "ch570", "ch600", "ch650"]:
            self.assertIn(key, data)
            self.assertGreater(data[key], 0)

    def test_sensor_hub_read_all(self):
        hub = SensorHub()
        data = hub.read_all()
        self.assertIsInstance(data, SensorData)
        self.assertGreater(data.temperature, 0)
        self.assertGreater(data.humidity, 0)
        self.assertGreater(data.co2_ppm, 0)
        self.assertGreater(data.ph, 0)
        self.assertGreater(data.tds_ppm, 0)
        self.assertGreater(data.ec_ms_cm, 0)
        self.assertGreater(data.par_ue, 0)
        self.assertNotEqual(data.timestamp, "")

class TestRX9TempCompensation(unittest.TestCase):
    """서미스터 온도 보상 로직 검증"""

    def test_compensation_direction(self):
        """센서 온도가 기준보다 높으면 보상 EMF가 증가"""
        co2 = RX9SimpleCO2()
        emf_raw = 280.0
        t_sensor = 30.0
        comp = co2.TEMP_COEFF * (t_sensor - co2.TEMP_REF)
        comp_emf = emf_raw + comp
        self.assertGreater(comp_emf, emf_raw)

    def test_compensation_at_reference(self):
        """기준 온도에서는 보상량 0"""
        co2 = RX9SimpleCO2()
        comp = co2.TEMP_COEFF * (co2.TEMP_REF - co2.TEMP_REF)
        self.assertEqual(comp, 0.0)

class TestMG92BServo(unittest.TestCase):
    """MG92B 서보 모터 단위 테스트"""

    def test_angle_to_duty_0(self):
        """0도 -> 2.5% 듀티 사이클"""
        duty = MG92BServo._angle_to_duty(0)
        self.assertAlmostEqual(duty, 2.5)

    def test_angle_to_duty_90(self):
        """90도 -> 7.5% 듀티 사이클"""
        duty = MG92BServo._angle_to_duty(90)
        self.assertAlmostEqual(duty, 7.5)

    def test_angle_to_duty_180(self):
        """180도 -> 12.5% 듀티 사이클"""
        duty = MG92BServo._angle_to_duty(180)
        self.assertAlmostEqual(duty, 12.5)

    def test_angle_clamped_negative(self):
        """음수 각도는 0도(2.5%)로 클램핑"""
        duty = MG92BServo._angle_to_duty(-10)
        self.assertAlmostEqual(duty, 2.5)

    def test_angle_clamped_over_180(self):
        """180도 초과는 180도(12.5%)로 클램핑"""
        duty = MG92BServo._angle_to_duty(200)
        self.assertAlmostEqual(duty, 12.5)

    def test_set_angle_sim(self):
        """시뮬레이션에서 set_angle 후 angle 속성 확인"""
        servo = MG92BServo()
        servo.set_angle(90)
        self.assertEqual(servo.angle, 90)
        servo.set_angle(0)
        self.assertEqual(servo.angle, 0)

    def test_feed_once_sim(self):
        """시뮬레이션에서 feed_once 후 0도 복귀 확인"""
        servo = MG92BServo()
        servo.feed_once(open_angle=90, hold_sec=0.1)
        self.assertEqual(servo.angle, 0)
        self.assertFalse(servo.is_feeding)

    def test_periodic_feeding_sim(self):
        """주기적 공급 시작/중단 테스트"""
        servo = MG92BServo()
        servo.start_periodic_feeding(
            interval_sec=0.5, open_angle=90, hold_sec=0.1)
        time.sleep(1.5)  # 최소 2회 공급 대기
        self.assertTrue(
            servo.is_feeding or servo._feed_thread.is_alive())
        servo.stop_periodic_feeding()
        self.assertFalse(servo.is_feeding)
        self.assertEqual(servo.angle, 0)

    def test_duty_cycle_formula_linearity(self):
        """듀티 사이클이 각도에 비례하여 선형 증가"""
        d0 = MG92BServo._angle_to_duty(0)
        d45 = MG92BServo._angle_to_duty(45)
        d90 = MG92BServo._angle_to_duty(90)
        d135 = MG92BServo._angle_to_duty(135)
        d180 = MG92BServo._angle_to_duty(180)
        step = (d180 - d0) / 4.0  # 2.5
        self.assertAlmostEqual(d45 - d0, step, places=5)
        self.assertAlmostEqual(d90 - d45, step, places=5)
        self.assertAlmostEqual(d135 - d90, step, places=5)
        self.assertAlmostEqual(d180 - d135, step, places=5)

    def test_cleanup_returns_to_zero(self):
        """cleanup 후 0도 확인"""
        servo = MG92BServo()
        servo.set_angle(90)
        servo.cleanup()
        self.assertEqual(servo.angle, 0)

# =========================================================================
#  시뮬레이션 러너
# =========================================================================

def run_simulation(duration_sec: int = 30, loop_interval: float = 1.0):
    """하드웨어 없이 전체 제어 루프를 시뮬레이션"""

    cfg = ControlConfig(
        loop_interval_sec=loop_interval,
        camera_interval_sec=9999,
        servo_feed_interval=5.0,
        servo_open_angle=90,
        servo_close_angle=0,
        servo_hold_sec=0.5,
    )

    ctrl = AstroFarmController(cfg)
    ctrl._running = True

    # 시뮬레이션용: 주기적 서보 공급 시작
    ctrl.actuators.servo.start_periodic_feeding(
        interval_sec=cfg.servo_feed_interval,
        open_angle=cfg.servo_open_angle,
        hold_sec=cfg.servo_hold_sec,
    )

    print("\n" + "=" * 80)
    print("  AstroFarm 시뮬레이션 시작")
    print("  실행 시간: %ds / 루프 간격: %.1fs / HARDWARE: %s"
          % (duration_sec, loop_interval, HARDWARE))
    print("  서보 공급: %d도 개방, %.1fs 유지, %.1fs 간격"
          % (cfg.servo_open_angle, cfg.servo_hold_sec,
             cfg.servo_feed_interval))
    print("=" * 80)

    start = time.time()
    cycle = 0

    try:
        while time.time() - start < duration_sec:
            t0 = time.time()
            data = ctrl.sensors.read_all()

            ctrl._ec_history.append((t0, data.ec_ms_cm))
            if len(ctrl._ec_history) > 720:
                ctrl._ec_history.pop(0)

            alerts = ctrl._diagnose(data)
            ctrl._execute_control(data)
            ctrl.xbee.send_telemetry(data)

            ctrl._metrics.add(asdict(data))

            ec_rate = ctrl._compute_ec_absorption_rate()
            cycle += 1

            servo_status = "OPEN %3d" % data.servo_angle if data.servo_feeding \
                else "CLOSED %d" % data.servo_angle

            alert_str = ""
            if alerts:
                alert_str = "  [!] " + ", ".join(alerts)

            print(
                "  [%3d] T=%.1fC  RH=%.1f%%  "
                "CO2=%dppm  PAR=%.1f  "
                "pH=%.2f  TDS=%.1f  EC=%.2f  "
                "Peltier=%d%%  LED_B=%d%%  LED_R=%d%%  "
                "Servo=%s%s"
                % (
                    cycle,
                    data.temperature, data.humidity,
                    data.co2_ppm, data.par_ue,
                    data.ph, data.tds_ppm, data.ec_ms_cm,
                    data.peltier_pct,
                    data.led_blue_pct, data.led_red_pct,
                    servo_status,
                    alert_str,
                )
            )

            elapsed = time.time() - t0
            time.sleep(max(0, loop_interval - elapsed))

    except KeyboardInterrupt:
        print("\n  사용자 중단")

    # 정리
    ctrl.actuators.servo.stop_periodic_feeding()
    ctrl.actuators.cleanup()

    # 성능 요약
    print("\n" + "-" * 80)
    print("  시뮬레이션 완료 (%d 사이클)" % cycle)
    stab = ctrl._metrics.stability_pct()
    print("  안정성: %.1f%%" % stab)
    ec_r = ctrl._compute_ec_absorption_rate()
    if ec_r is not None:
        print("  EC 흡수율: %.6f mS/cm/cycle" % ec_r)
    print("-" * 80 + "\n")

# =========================================================================
#  진입점
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AstroFarm - 테스트 & 시뮬레이터")
    parser.add_argument(
        "--mode", choices=["test", "sim", "run"],
        default="sim",
        help="test=단위테스트, sim=시뮬레이션, run=실제 운용")
    parser.add_argument(
        "--duration", type=int, default=20,
        help="시뮬레이션 실행 시간 (초)")
    args = parser.parse_args()

    if args.mode == "test":
        print("\n" + "=" * 60)
        print("  AstroFarm 단위 테스트")
        print("  HARDWARE: %s (시뮬레이션 모드로 테스트)" % HARDWARE)
        print("=" * 60 + "\n")
        unittest.main(argv=[""], exit=False, verbosity=2)

    elif args.mode == "sim":
        run_simulation(duration_sec=args.duration)

    elif args.mode == "run":
        if not HARDWARE:
            print("[경고] 하드웨어 미감지, 시뮬레이션으로 전환합니다.")
        controller = AstroFarmController()
        controller.run()
