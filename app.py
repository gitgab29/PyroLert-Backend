from flask import Flask, jsonify, request

from inference.features import OnlineFeatureState
from inference.predictor import Predictor

app = Flask(__name__)

_state = OnlineFeatureState()
_predictor = Predictor()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/status")
def status():
    return jsonify({
        "n_samples": _state.n_samples,
        "warmed_up": _state.n_samples > 30,
    })


@app.post("/reset")
def reset():
    _state.reset()
    return jsonify({"status": "reset"})


@app.post("/infer")
def infer():
    raw = request.get_json(force=True)

    if not raw.get("pms_valid", 1):
        return jsonify({"prediction": "NONE", "reason": "pms_invalid"})

    feats = _state.update(raw)

    if feats is None:
        return jsonify({"prediction": "NONE", "reason": "warmup", "n": _state.n_samples})

    result = _predictor.predict(feats)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
