"""
Smoke test — run with: python smoke_test.py
Verifies imports, warmup logic, feature vector shapes, and inference response shape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from inference.features import OnlineFeatureState
from inference.predictor import Predictor

FAKE_ROW = {
    "mq135": 120.0, "mq4": 80.0, "mq5": 60.0,  # mq5 col = MQ9 sensor
    "mq6": 55.0, "mq7": 45.0, "mq8": 40.0,
    "temperature_c": 27.5, "humidity_pct": 55.0,
    "pm1_0_atm": 5.0, "pm2_5_atm": 8.0, "pm10_atm": 12.0,
    "pcnt_0_3um": 1200.0, "pcnt_0_5um": 400.0, "pcnt_1_0um": 80.0,
    "pcnt_2_5um": 10.0, "pcnt_5_0um": 2.0, "pcnt_10um": 0.5,
    "pms_valid": 1, "warmed_up": 1,
}

print("Loading models...", flush=True)
predictor = Predictor()
print("  Stage-1 features:", len(predictor._s1_feats))
print("  Stage-2 features:", len(predictor._s2_feats))
assert len(predictor._s1_feats) == 24
assert len(predictor._s2_feats) == 77

state = OnlineFeatureState()

# --- Warmup check: first 30 calls must return None ---
for i in range(30):
    result = state.update(FAKE_ROW)
    assert result is None, f"Expected None during warmup at step {i+1}, got {result}"

assert state.n_samples == 30
print(f"Warmup OK: {state.n_samples} samples collected, still in warmup")

# --- 31st call must produce features ---
feats = state.update(FAKE_ROW)
assert feats is not None, "31st update should return feature dict"
print(f"Feature dict OK: {len(feats)} features at n_samples={state.n_samples}")

# --- Verify all S1 and S2 feature names are present ---
missing_s1 = [n for n in predictor._s1_feats if n not in feats]
assert not missing_s1, f"Missing S1 feature keys: {missing_s1}"

missing_s2 = [n for n in predictor._s2_feats if n not in feats]
assert not missing_s2, f"Missing S2 feature keys: {missing_s2}"

print("Feature name coverage OK")

# --- Run inference ---
result = predictor.predict(feats)
assert "prediction" in result
assert result["prediction"] in ("NONE", "A", "C", "K")
assert "s1_proba" in result
print(f"Inference OK: prediction={result['prediction']}, s1_proba={result['s1_proba']:.4f}")
if result.get("s2_proba"):
    print(f"  s2_proba={result['s2_proba']}, confidence={result['confidence']:.4f}")

# --- Reset ---
state.reset()
assert state.n_samples == 0
print("Reset OK: n_samples=0")

print("\nAll checks passed.")
