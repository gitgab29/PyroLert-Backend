import json
import pathlib

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"

S1_THRESHOLD = 0.60
S2_CONFIDENCE = 0.40
EPS = 1e-6

S2_SEEDS = [42, 0, 1, 7, 13, 17, 99]
S2_CLASSES = ["A", "C", "K"]  # ONNX encoder: A→0, C→1, K→2


def _cat(f: str) -> str:
    if f.startswith(("pm1_0_atm", "pm2_5_atm", "pm10_atm", "pcnt_")):
        return "pms"
    if f.startswith("ratio_pm") or f.startswith("ratio_pcnt"):
        return "pms"
    if f.startswith("mq") or f.startswith("ratio_mq"):
        return "mos"
    if f in ("temperature_c", "humidity_pct", "warmed_up"):
        return "env"
    return "unknown"


class Predictor:
    def __init__(self):
        with open(MODELS_DIR / "feature_names.json") as fh:
            self._feat_names = json.load(fh)

        # Feature name lists for each stage (preserves feature_names.json ordering)
        self._s1_feats = (
            [f for f in self._feat_names if _cat(f) == "pms"]
            + [f for f in self._feat_names if _cat(f) == "env"]
        )
        self._s2_base_feats = (
            [f for f in self._feat_names if _cat(f) == "mos"]
            + [f for f in self._feat_names if _cat(f) == "env"]
        )

        assert len(self._s1_feats) == 24, (
            f"Stage-1 expects 24 features, got {len(self._s1_feats)}"
        )
        assert len(self._s2_base_feats) == 72, (
            f"Stage-2 base expects 72 features, got {len(self._s2_base_feats)}"
        )

        # Load Stage-1 LightGBM booster
        self._s1 = lgb.Booster(
            model_file=str(MODELS_DIR / "two_stage_partitioned" / "stage1" / "model.txt")
        )

        # Load Stage-2 scaler and 7-seed ONNX ensemble
        self._scaler = joblib.load(MODELS_DIR / "two_stage_ensemble" / "scaler.joblib")
        self._s2_sessions = [
            ort.InferenceSession(
                str(MODELS_DIR / "two_stage_ensemble" / f"mlp_seed_{s}.onnx")
            )
            for s in S2_SEEDS
        ]

        # Label encoder (loaded as required; not used for ONNX decoding)
        self._label_enc = joblib.load(MODELS_DIR / "label_encoder.joblib")

    def predict(self, feat: dict) -> dict:
        # --- Stage 1 ---
        x_s1 = np.array([feat[f] for f in self._s1_feats], dtype=np.float64).reshape(1, -1)
        s1_proba = float(self._s1.predict(x_s1)[0])

        smoke_detected = s1_proba >= S1_THRESHOLD

        if not smoke_detected:
            return {"prediction": "NONE", "smoke_detected": smoke_detected,
                    "s1_proba": s1_proba, "s2_proba": None, "confidence": None}

        # --- Stage 2 base features (72) ---
        x_s2_base = np.array([feat[f] for f in self._s2_base_feats], dtype=np.float32)

        # 5 extra cross-ratio features (EPS avoids div-by-zero)
        extra = np.array([
            feat["mq135"]      / (feat["mq5"]      + EPS),  # ratio_mq135_mq9
            feat["mq7"]        / (feat["mq6"]      + EPS),  # ratio_mq7_mq6
            feat["mq4"]        / (feat["mq6"]      + EPS),  # ratio_mq4_mq6
            feat["mq135_ewma"] / (feat["mq5_ewma"] + EPS),  # ratio_mq135_mq9_ewma
            feat["mq7_ewma"]   / (feat["mq6_ewma"] + EPS),  # ratio_mq7_mq6_ewma
        ], dtype=np.float32)

        x_s2 = np.concatenate([x_s2_base, extra]).reshape(1, -1)
        assert x_s2.shape[1] == 77, f"Stage-2 expects 77 features, got {x_s2.shape[1]}"

        x_s2_scaled = self._scaler.transform(x_s2).astype(np.float32)

        # Average softmax probabilities across all 7 ONNX models
        probas = []
        for sess in self._s2_sessions:
            inp_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {inp_name: x_s2_scaled})
            # output[1] has shape (n_samples, 3); use [0] for our single row
            probas.append(outputs[1][0])

        avg_proba = np.mean(probas, axis=0)  # shape (3,)
        confidence = float(np.max(avg_proba))
        s2_proba = {cls: round(float(avg_proba[i]), 6) for i, cls in enumerate(S2_CLASSES)}

        if confidence < S2_CONFIDENCE:
            return {
                "prediction": "NONE",
                "smoke_detected": smoke_detected,
                "s1_proba": s1_proba,
                "s2_proba": s2_proba,
                "confidence": confidence,
                "reason": "low_confidence",
            }

        prediction = S2_CLASSES[int(np.argmax(avg_proba))]
        return {
            "prediction": prediction,
            "smoke_detected": smoke_detected,
            "s1_proba": s1_proba,
            "s2_proba": s2_proba,
            "confidence": confidence,
        }
