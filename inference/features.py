from collections import deque
import numpy as np

ALPHA = 0.005
RING_LEN = 31
PM_MED_LEN = 5
WARMUP_N = 30

DYNAMIC_COLS = ["mq135", "mq4", "mq5", "mq6", "mq7", "mq8", "pm2_5_atm"]
PM_COLS = ["pm1_0_atm", "pm2_5_atm", "pm10_atm"]


class OnlineFeatureState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_samples = 0
        self._pm_buf = {c: deque(maxlen=PM_MED_LEN) for c in PM_COLS}
        self._ring = {c: deque(maxlen=RING_LEN) for c in DYNAMIC_COLS}
        self._ewma = {c: None for c in DYNAMIC_COLS}

    def update(self, raw: dict):
        """
        Process one valid (pms_valid=1) sensor row.
        Returns 93-feature dict once warmed up, else None.
        """
        # Step 1: PM causal median filter — replace raw PM with rolling median
        for c in PM_COLS:
            self._pm_buf[c].append(float(raw[c]))
        pm = {c: float(np.median(list(self._pm_buf[c]))) for c in PM_COLS}

        # Step 2: Update ring buffers and EWMA for each dynamic channel
        for c in DYNAMIC_COLS:
            v = pm["pm2_5_atm"] if c == "pm2_5_atm" else float(raw[c])
            self._ring[c].append(v)
            if self._ewma[c] is None:
                self._ewma[c] = v
            else:
                self._ewma[c] = ALPHA * v + (1.0 - ALPHA) * self._ewma[c]

        self.n_samples += 1
        if self.n_samples <= WARMUP_N:
            return None

        return self._build(raw, pm)

    def _build(self, raw: dict, pm: dict) -> dict:
        f = {}

        # Raw kept (18)
        for col in ("mq135", "mq4", "mq5", "mq6", "mq7", "mq8"):
            f[col] = float(raw[col])
        f["temperature_c"] = float(raw["temperature_c"])
        f["humidity_pct"] = float(raw["humidity_pct"])
        f["pm1_0_atm"] = pm["pm1_0_atm"]
        f["pm2_5_atm"] = pm["pm2_5_atm"]
        f["pm10_atm"] = pm["pm10_atm"]
        for col in ("pcnt_0_3um", "pcnt_0_5um", "pcnt_1_0um",
                    "pcnt_2_5um", "pcnt_5_0um", "pcnt_10um"):
            f[col] = float(raw[col])
        f["warmed_up"] = float(raw["warmed_up"])

        # Per dynamic target (7 channels × 10 features = 70)
        # Ring buffer is full (len=31) when inference runs, so all indices are safe.
        for col in DYNAMIC_COLS:
            buf = list(self._ring[col])
            cur = buf[-1]
            ewma = self._ewma[col]

            d1s = cur - buf[-2]   # 1 sample back
            d3s = cur - buf[-4]   # 3 samples back
            d10s = cur - buf[-11] # 10 samples back
            d30s = cur - buf[-31] # 30 samples back

            if col == "pm2_5_atm":
                d1s = float(np.clip(d1s, -500.0, 500.0))

            # slope_Ws = (current - value[t-(W-1)]) / (W-1)
            mean_10s = float(np.mean(buf[-10:]))
            slope_10s = (cur - buf[-10]) / 9.0   # W=10: buf[-10] = value[t-9]

            mean_30s = float(np.mean(buf[-30:]))
            slope_30s = (cur - buf[-30]) / 29.0  # W=30: buf[-30] = value[t-29]

            f[f"{col}_d1s"] = float(d1s)
            f[f"{col}_d3s"] = float(d3s)
            f[f"{col}_d10s"] = float(d10s)
            f[f"{col}_d30s"] = float(d30s)
            f[f"{col}_mean_10s"] = mean_10s
            f[f"{col}_slope_10s"] = float(slope_10s)
            f[f"{col}_mean_30s"] = mean_30s
            f[f"{col}_slope_30s"] = float(slope_30s)
            f[f"{col}_ewma"] = float(ewma)
            f[f"{col}_ewma_delta"] = float(cur - ewma)

        # Ratios (5) — use filtered PM values where applicable
        f["ratio_mq7_mq4"] = f["mq7"] / (f["mq4"] + 1.0)
        f["ratio_mq8_mq9"] = f["mq8"] / (f["mq5"] + 1.0)        # mq5 col = MQ9 sensor
        f["ratio_mq135_mq4"] = f["mq135"] / (f["mq4"] + 1.0)
        f["ratio_pm10_pm25"] = pm["pm10_atm"] / (pm["pm2_5_atm"] + 1.0)
        f["ratio_pcnt_03_25"] = f["pcnt_0_3um"] / (f["pcnt_2_5um"] + 1.0)

        return f
