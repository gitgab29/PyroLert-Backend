from collections import deque
import numpy as np

ALPHA = 0.005
MQ_RING_LEN = 121   # needs 121 entries for d120s (buf[-121])
PM_RING_LEN = 31    # needs 31 entries for d30s (buf[-31])
PM_MED_LEN = 5
WARMUP_N = 30
EPS = 1e-6

MQ_COLS = ["mq135", "mq4", "mq5", "mq6", "mq7", "mq8"]
PM_COLS = ["pm1_0_atm", "pm2_5_atm", "pm10_atm"]


class OnlineFeatureState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_samples = 0
        self._pm_buf = {c: deque(maxlen=PM_MED_LEN) for c in PM_COLS}
        self._mq_ring = {c: deque(maxlen=MQ_RING_LEN) for c in MQ_COLS}
        self._pm_ring = deque(maxlen=PM_RING_LEN)
        self._mq_ewma = {c: None for c in MQ_COLS}
        self._pm_ewma = None

    def update(self, raw: dict):
        """
        Process one valid (pms_valid=1) sensor row.
        Returns feature dict once warmed up (> 30 samples), else None.
        """
        # PM causal median filter
        for c in PM_COLS:
            self._pm_buf[c].append(float(raw[c]))
        pm = {c: float(np.median(list(self._pm_buf[c]))) for c in PM_COLS}

        # MQ ring buffers and EWMA (alpha=0.005)
        for c in MQ_COLS:
            v = float(raw[c])
            self._mq_ring[c].append(v)
            if self._mq_ewma[c] is None:
                self._mq_ewma[c] = v
            else:
                self._mq_ewma[c] = ALPHA * v + (1.0 - ALPHA) * self._mq_ewma[c]

        # pm2_5_atm ring and EWMA for S1 derived features
        pm25 = pm["pm2_5_atm"]
        self._pm_ring.append(pm25)
        if self._pm_ewma is None:
            self._pm_ewma = pm25
        else:
            self._pm_ewma = ALPHA * pm25 + (1.0 - ALPHA) * self._pm_ewma

        self.n_samples += 1
        if self.n_samples <= WARMUP_N:
            return None

        return self._build(raw, pm)

    def _build(self, raw: dict, pm: dict) -> dict:
        f = {}

        # ── S1 features — PMS + env (24 total, unchanged) ────────────────────
        f["pm1_0_atm"] = pm["pm1_0_atm"]
        f["pm2_5_atm"] = pm["pm2_5_atm"]
        f["pm10_atm"]  = pm["pm10_atm"]
        for col in ("pcnt_0_3um", "pcnt_0_5um", "pcnt_1_0um",
                    "pcnt_2_5um", "pcnt_5_0um", "pcnt_10um"):
            f[col] = float(raw[col])
        f["temperature_c"] = float(raw["temperature_c"])
        f["humidity_pct"]  = float(raw["humidity_pct"])
        f["warmed_up"]     = float(raw["warmed_up"])

        # pm2_5_atm derived (10 features) — pm_ring has >= 31 entries after warmup
        pm_buf   = list(self._pm_ring)
        pm25_cur = pm_buf[-1]
        pm25_ewma = self._pm_ewma

        f["pm2_5_atm_d1s"]       = float(np.clip(pm25_cur - pm_buf[-2],  -500.0, 500.0))
        f["pm2_5_atm_d3s"]       = float(pm25_cur - pm_buf[-4])
        f["pm2_5_atm_d10s"]      = float(pm25_cur - pm_buf[-11])
        f["pm2_5_atm_d30s"]      = float(pm25_cur - pm_buf[-31])
        f["pm2_5_atm_mean_10s"]  = float(np.mean(pm_buf[-10:]))
        f["pm2_5_atm_slope_10s"] = float((pm25_cur - pm_buf[-10]) / 9.0)
        f["pm2_5_atm_mean_30s"]  = float(np.mean(pm_buf[-30:]))
        f["pm2_5_atm_slope_30s"] = float((pm25_cur - pm_buf[-30]) / 29.0)
        f["pm2_5_atm_ewma"]      = float(pm25_ewma)
        f["pm2_5_atm_ewma_delta"] = float(pm25_cur - pm25_ewma)

        f["ratio_pm10_pm25"]   = pm["pm10_atm"] / (pm["pm2_5_atm"] + 1.0)
        f["ratio_pcnt_03_25"]  = f["pcnt_0_3um"] / (f["pcnt_2_5um"] + 1.0)

        # ── S2 features — MQ per-channel, baseline-invariant ─────────────────
        # mq5 column = MQ9 sensor (firmware mislabel — never call it MQ5)
        for col in MQ_COLS:
            buf  = list(self._mq_ring[col])
            cur  = buf[-1]
            ewma = self._mq_ewma[col]
            n    = len(buf)

            f[f"{col}_ewma_delta"]        = float(cur - ewma)
            f[f"{col}_d1s"]               = float(cur - buf[-2])
            f[f"{col}_d3s"]               = float(cur - buf[-4])
            f[f"{col}_d10s"]              = float(cur - buf[-11])
            f[f"{col}_d30s"]              = float(cur - buf[-31])
            f[f"{col}_mean10_above_ewma"] = float(np.mean(buf[-10:]) - ewma)
            f[f"{col}_mean30_above_ewma"] = float(np.mean(buf[-30:]) - ewma)
            f[f"{col}_d60s"]              = float(cur - buf[-61])  if n >= 61  else 0.0
            f[f"{col}_d120s"]             = float(cur - buf[-121]) if n >= 121 else 0.0
            f[f"{col}_mean60_above_ewma"] = float(np.mean(buf[-60:])  - ewma)
            f[f"{col}_mean120_above_ewma"] = float(np.mean(buf[-120:]) - ewma)

        # EWMA-delta ratios (8) — EPS guards against near-zero denominator
        mq135_ed = f["mq135_ewma_delta"]
        mq4_ed   = f["mq4_ewma_delta"]
        mq5_ed   = f["mq5_ewma_delta"]   # mq5 col = MQ9 sensor
        mq6_ed   = f["mq6_ewma_delta"]
        mq7_ed   = f["mq7_ewma_delta"]
        mq8_ed   = f["mq8_ewma_delta"]

        f["ewmadelta_ratio_mq135_mq9"] = mq135_ed / (mq5_ed  + EPS)
        f["ewmadelta_ratio_mq7_mq6"]   = mq7_ed   / (mq6_ed  + EPS)
        f["ewmadelta_ratio_mq4_mq6"]   = mq4_ed   / (mq6_ed  + EPS)
        f["ewmadelta_ratio_mq7_mq9"]   = mq7_ed   / (mq5_ed  + EPS)
        f["ewmadelta_ratio_mq135_mq4"] = mq135_ed / (mq4_ed  + EPS)
        f["ewmadelta_ratio_mq8_mq6"]   = mq8_ed   / (mq6_ed  + EPS)
        f["ewmadelta_ratio_mq8_mq9"]   = mq8_ed   / (mq5_ed  + EPS)
        f["ewmadelta_ratio_mq6_mq4"]   = mq6_ed   / (mq4_ed  + EPS)

        return f
