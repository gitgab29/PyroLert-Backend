import json
import pathlib

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models" / "two_stage_ec2"

S1_THRESHOLD  = 0.25
CONFIDENCE_GATE = 0.0   # no gating — meta-learner handles class separation
S2_SEEDS      = [42, 0, 1, 7, 13, 17, 99]
S2_CLASSES    = ["A", "C", "K"]

# Stage-1 expects exactly these 24 PMS+env features in this order.
# Order matches the original feature_names.json (PMS first, then env) which is
# what the S1 model was trained on. Unchanged from prior backend.
_S1_FEATS = [
    "pm1_0_atm", "pm2_5_atm", "pm10_atm",
    "pcnt_0_3um", "pcnt_0_5um", "pcnt_1_0um", "pcnt_2_5um", "pcnt_5_0um", "pcnt_10um",
    "pm2_5_atm_d1s", "pm2_5_atm_d3s", "pm2_5_atm_d10s", "pm2_5_atm_d30s",
    "pm2_5_atm_mean_10s", "pm2_5_atm_slope_10s",
    "pm2_5_atm_mean_30s", "pm2_5_atm_slope_30s",
    "pm2_5_atm_ewma", "pm2_5_atm_ewma_delta",
    "ratio_pm10_pm25", "ratio_pcnt_03_25",
    "temperature_c", "humidity_pct", "warmed_up",
]


class _MLP(nn.Module):
    def __init__(self, n_in, hidden, n_layers, n_out=3):
        super().__init__()
        layers = []
        prev = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(), nn.Dropout(0.0)]
            prev = hidden
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_mlp(path: pathlib.Path) -> _MLP:
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    n_in   = sd["net.0.weight"].shape[1]
    hidden = sd["net.0.weight"].shape[0]
    n_linear = sum(1 for k, v in sd.items() if k.endswith(".weight") and v.dim() == 2)
    n_layers = n_linear - 1   # subtract the final output layer
    mlp = _MLP(n_in, hidden, n_layers)
    mlp.load_state_dict(sd)
    mlp.eval()
    return mlp


class Predictor:
    def __init__(self):
        # Stage-1 LightGBM gate
        self._s1 = lgb.Booster(model_file=str(MODELS_DIR / "stage1/model.txt"))
        self._s1_feats = _S1_FEATS

        # Stage-2 LightGBM
        self._s2_lgbm = lgb.Booster(model_file=str(MODELS_DIR / "stage2/lgbm_model.txt"))

        # Stage-2 MLP ensemble (PyTorch, CPU, eval mode)
        self._s2_mlp_models = [
            _load_mlp(MODELS_DIR / f"stage2/mlp_seed_{s}.pt")
            for s in S2_SEEDS
        ]

        # Scaler applied to MLP input only
        self._scaler = joblib.load(MODELS_DIR / "stage2/scaler.joblib")

        # Meta-learner (LogisticRegression stacking LGBM + MLP probas)
        self._meta = joblib.load(MODELS_DIR / "stage2/meta_learner.joblib")

        # Label encoder (kept for completeness; classes resolved via S2_CLASSES constant)
        self._le = joblib.load(MODELS_DIR / "label_encoder.joblib")

        # S2 feature ordering — 77 names, exact order the models were trained on
        with open(MODELS_DIR / "stage2/feature_names.json") as fh:
            names_cfg = json.load(fh)
        self._s2_feats = names_cfg["s2_all_tabular_features"]

        assert len(self._s1_feats) == 24, (
            f"Stage-1 expects 24 features, got {len(self._s1_feats)}"
        )
        assert len(self._s2_feats) == 77, (
            f"Stage-2 expects 77 features, got {len(self._s2_feats)}"
        )

    def predict(self, feat: dict) -> dict:
        # ── Stage 1 ──────────────────────────────────────────────────────────
        x_s1 = np.array([feat[f] for f in self._s1_feats], dtype=np.float64).reshape(1, -1)
        s1_proba = float(self._s1.predict(x_s1)[0])

        if s1_proba < S1_THRESHOLD:
            return {"prediction": "NONE", "smoke_detected": False,
                    "s1_proba": s1_proba, "s2_proba": None, "confidence": None}

        # ── Stage 2 ──────────────────────────────────────────────────────────
        x_s2 = np.array([feat[f] for f in self._s2_feats], dtype=np.float32).reshape(1, -1)

        # LightGBM proba — shape (1, 3)
        lgbm_proba = self._s2_lgbm.predict(x_s2)

        # MLP proba — average softmax over 7 seeds, shape (1, 3)
        x_s2_scaled = self._scaler.transform(x_s2).astype(np.float32)
        xt = torch.tensor(x_s2_scaled)
        mlp_probas = []
        for mlp in self._s2_mlp_models:
            with torch.no_grad():
                mlp_probas.append(torch.softmax(mlp(xt), dim=1).numpy())
        mlp_proba = np.mean(mlp_probas, axis=0)  # (1, 3)

        # Stack LGBM + MLP probas and run meta-learner
        stacked    = np.hstack([lgbm_proba, mlp_proba])          # (1, 6)
        meta_proba = self._meta.predict_proba(stacked)[0]         # (3,)

        confidence = float(np.max(meta_proba))
        s2_proba   = {cls: round(float(meta_proba[i]), 6)
                      for i, cls in enumerate(S2_CLASSES)}
        prediction = S2_CLASSES[int(np.argmax(meta_proba))]

        # CONFIDENCE_GATE = 0.0 — no explicit gating; meta-learner handles separation
        return {"prediction": prediction, "smoke_detected": True,
                "s1_proba": s1_proba, "s2_proba": s2_proba, "confidence": confidence}
