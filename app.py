import time

from flask import Flask, jsonify, request, Response

from inference.features import OnlineFeatureState
from inference.predictor import Predictor

app = Flask(__name__)

_state     = OnlineFeatureState()
_predictor = Predictor()

# Last-reading state — updated on every POST /infer
_last_seen   = None   # epoch float of last POST
_last_raw    = None   # last raw sensor dict (for dashboard display)
_last_result = None   # last prediction dict


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/")
def dashboard():
    return Response(_DASHBOARD_HTML, content_type="text/html")


# ── Health / status ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/status")
def status():
    now = time.time()
    return jsonify({
        "n_samples":          _state.n_samples,
        "warmed_up":          _state.n_samples > 30,
        "device_online":      _last_seen is not None and (now - _last_seen) < 5,
        "seconds_since_last": round(now - _last_seen, 1) if _last_seen else None,
    })


@app.get("/latest")
def latest():
    now = time.time()
    online = _last_seen is not None and (now - _last_seen) < 5
    resp = {
        "device_online":      online,
        "seconds_since_last": round(now - _last_seen, 1) if _last_seen else None,
        "n_samples":          _state.n_samples,
        "warmed_up":          _state.n_samples > 30,
        "prediction":         None,
        "s1_proba":           None,
        "s2_proba":           None,
        "confidence":         None,
        "reason":             None,
        "sensors":            None,
    }
    if _last_result:
        resp.update(_last_result)
    if _last_raw:
        resp["sensors"] = {k: _last_raw.get(k) for k in (
            "mq135", "mq4", "mq5", "mq6", "mq7", "mq8",
            "temperature_c", "humidity_pct",
            "pm1_0_atm", "pm2_5_atm", "pm10_atm",
            "pms_valid",
        )}
    return jsonify(resp)


# ── Control ──────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    global _last_seen, _last_raw, _last_result
    _state.reset()
    _last_seen = _last_raw = _last_result = None
    return jsonify({"status": "reset"})


# ── Inference ────────────────────────────────────────────────────────────────

@app.post("/infer")
def infer():
    global _last_seen, _last_raw, _last_result
    raw = request.get_json(force=True)

    _last_seen = time.time()
    _last_raw  = raw

    if not raw.get("pms_valid", 1):
        result = {"prediction": "NONE", "reason": "pms_invalid"}
        _last_result = result
        return jsonify(result)

    feats = _state.update(raw)

    if feats is None:
        result = {"prediction": "NONE", "reason": "warmup", "n": _state.n_samples}
        _last_result = result
        return jsonify(result)

    result = _predictor.predict(feats)
    _last_result = result
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


# ── Embedded dashboard ───────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyroLert Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;min-height:100vh;padding:20px 24px 40px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}
.logo{font-size:1.2rem;font-weight:700;letter-spacing:-.02em}
.logo em{color:#f97316;font-style:normal}

/* Device badge */
.badge{padding:5px 14px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.04em;display:flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.badge-wait{background:#1e2330;color:#64748b}.badge-wait .dot{background:#64748b}
.badge-on  {background:#14532d;color:#4ade80}.badge-on   .dot{background:#4ade80;animation:pulse 1.5s infinite}
.badge-off {background:#7f1d1d;color:#f87171}.badge-off  .dot{background:#f87171}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Grid */
.grid{display:grid;gap:14px;grid-template-columns:1fr 1fr}
@media(max-width:600px){.grid{grid-template-columns:1fr}}

/* Cards */
.card{background:#161b27;border:1px solid #1e2d40;border-radius:14px;padding:20px}
.label{font-size:.66rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#4b5a6e;margin-bottom:14px}

/* Prediction card */
.pred-card{grid-column:1/-1;text-align:center;padding:28px 24px 24px}
.pred-cls{font-size:5.5rem;font-weight:900;letter-spacing:-.04em;line-height:1;transition:color .25s}
.pred-name{font-size:.9rem;color:#94a3b8;margin-top:6px;min-height:1.2em}
.pred-reason{font-size:.72rem;color:#4b5a6e;margin-top:4px;min-height:1em}
.cls-none{color:#4b5a6e} .cls-A{color:#f97316} .cls-C{color:#60a5fa} .cls-K{color:#fbbf24}

/* Stats row under prediction */
.stats{display:flex;justify-content:center;gap:28px;margin-top:20px;flex-wrap:wrap}
.stat{text-align:center}
.stat-v{font-size:1.4rem;font-weight:700}
.stat-l{font-size:.65rem;color:#4b5a6e;margin-top:3px;text-transform:uppercase;letter-spacing:.06em}
.muted{color:#4b5a6e}

/* Stage 2 bars */
.bars{display:flex;flex-direction:column;gap:10px}
.bar-row{display:flex;align-items:center;gap:10px;font-size:.83rem}
.bar-lbl{width:14px;font-weight:700}
.bar-bg{flex:1;height:9px;background:#1e2d40;border-radius:5px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;transition:width .4s ease}
.bar-pct{width:38px;text-align:right;font-size:.72rem;color:#64748b}
.fill-A{background:#f97316} .fill-C{background:#60a5fa} .fill-K{background:#fbbf24}

/* Sensor grid */
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.si-name{font-size:.66rem;color:#4b5a6e;text-transform:uppercase;letter-spacing:.06em}
.si-val{font-size:1.1rem;font-weight:600;margin-top:3px}
.si-unit{font-size:.68rem;color:#4b5a6e}

.pms-row{display:flex;align-items:center;gap:8px;margin-top:14px;font-size:.8rem;color:#94a3b8}

.key{font-size:.78rem;color:#94a3b8;line-height:1.9;margin-top:14px}

.footer{font-size:.68rem;color:#1e2d40;text-align:right;margin-top:18px}
</style>
</head>
<body>

<header>
  <div class="logo"><em>Pyro</em>Lert Monitor</div>
  <div class="badge badge-wait" id="badge">
    <span class="dot"></span>
    <span id="badge-txt">WAITING FOR DEVICE</span>
  </div>
</header>

<div class="grid">

  <!-- Prediction -->
  <div class="card pred-card">
    <div class="label">Fire Classification</div>
    <div class="pred-cls cls-none" id="pred">—</div>
    <div class="pred-name" id="pred-name">No data yet</div>
    <div class="pred-reason" id="pred-reason"></div>
    <div class="stats">
      <div class="stat"><div class="stat-v muted" id="s1">—</div><div class="stat-l">Smoke prob (S1)</div></div>
      <div class="stat"><div class="stat-v muted" id="conf">—</div><div class="stat-l">Confidence (S2)</div></div>
      <div class="stat"><div class="stat-v muted" id="n-samples">0</div><div class="stat-l">Samples</div></div>
      <div class="stat"><div class="stat-v muted" id="warmup">—</div><div class="stat-l">Status</div></div>
    </div>
  </div>

  <!-- Stage 2 breakdown -->
  <div class="card">
    <div class="label">Stage 2 — Class Probabilities</div>
    <div class="bars">
      <div class="bar-row">
        <span class="bar-lbl cls-A">A</span>
        <div class="bar-bg"><div class="bar-fill fill-A" id="bar-A" style="width:0%"></div></div>
        <span class="bar-pct" id="pct-A">—</span>
      </div>
      <div class="bar-row">
        <span class="bar-lbl cls-C">C</span>
        <div class="bar-bg"><div class="bar-fill fill-C" id="bar-C" style="width:0%"></div></div>
        <span class="bar-pct" id="pct-C">—</span>
      </div>
      <div class="bar-row">
        <span class="bar-lbl cls-K">K</span>
        <div class="bar-bg"><div class="bar-fill fill-K" id="bar-K" style="width:0%"></div></div>
        <span class="bar-pct" id="pct-K">—</span>
      </div>
    </div>
    <div class="key">
      <span class="cls-A">A</span> Ordinary combustibles (wood, paper)<br>
      <span class="cls-C">C</span> Electrical fire<br>
      <span class="cls-K">K</span> Cooking oil / kitchen
    </div>
  </div>

  <!-- Gas sensors -->
  <div class="card">
    <div class="label">Gas Sensors (MOS)</div>
    <div class="sg">
      <div><div class="si-name">MQ135</div><div class="si-val" id="v-mq135">—</div></div>
      <div><div class="si-name">MQ4</div><div class="si-val" id="v-mq4">—</div></div>
      <div><div class="si-name">MQ9 <span style="color:#1e2d40">(mq5)</span></div><div class="si-val" id="v-mq5">—</div></div>
      <div><div class="si-name">MQ6</div><div class="si-val" id="v-mq6">—</div></div>
      <div><div class="si-name">MQ7</div><div class="si-val" id="v-mq7">—</div></div>
      <div><div class="si-name">MQ8</div><div class="si-val" id="v-mq8">—</div></div>
    </div>
    <div class="label" style="margin-top:18px">Environment</div>
    <div class="sg" style="grid-template-columns:1fr 1fr">
      <div>
        <div class="si-name">Temperature</div>
        <div><span class="si-val" id="v-temp">—</span> <span class="si-unit">°C</span></div>
      </div>
      <div>
        <div class="si-name">Humidity</div>
        <div><span class="si-val" id="v-hum">—</span> <span class="si-unit">%</span></div>
      </div>
    </div>
  </div>

  <!-- Particulate -->
  <div class="card">
    <div class="label">Particulate — PMS7003 ATM</div>
    <div class="sg">
      <div>
        <div class="si-name">PM 1.0</div>
        <div><span class="si-val" id="v-pm1">—</span> <span class="si-unit">µg/m³</span></div>
      </div>
      <div>
        <div class="si-name">PM 2.5</div>
        <div><span class="si-val" id="v-pm25">—</span> <span class="si-unit">µg/m³</span></div>
      </div>
      <div>
        <div class="si-name">PM 10</div>
        <div><span class="si-val" id="v-pm10">—</span> <span class="si-unit">µg/m³</span></div>
      </div>
    </div>
    <div class="pms-row">
      <span class="dot" id="pms-dot" style="background:#1e2d40"></span>
      <span id="pms-txt">PMS status unknown</span>
    </div>
  </div>

</div>

<div class="footer" id="tick">—</div>

<script>
const NAMES  = {NONE:'No fire detected', A:'Ordinary combustibles', C:'Electrical fire', K:'Cooking oil'};
const COLORS = {NONE:'cls-none', A:'cls-A', C:'cls-C', K:'cls-K'};

const $ = id => document.getElementById(id);
const set = (id, v) => { $(id).textContent = v; };
const fmt = (v, d=3) => v == null ? '—' : Number(v).toFixed(d);

async function poll() {
  let d;
  try {
    const r = await fetch('/latest');
    if (!r.ok) return;
    d = await r.json();
  } catch { return; }

  // Badge
  const badge = $('badge');
  const secs  = d.seconds_since_last;
  if (d.device_online) {
    badge.className = 'badge badge-on';
    set('badge-txt', `ONLINE  ${secs != null ? secs.toFixed(1)+'s ago' : ''}`);
  } else if (secs == null) {
    badge.className = 'badge badge-wait';
    set('badge-txt', 'WAITING FOR DEVICE');
  } else {
    badge.className = 'badge badge-off';
    set('badge-txt', `OFFLINE — ${secs.toFixed(0)}s since last packet`);
  }

  // Prediction
  const pred  = d.prediction || 'NONE';
  const predEl = $('pred');
  predEl.textContent = pred === 'NONE' && !d.prediction ? '—' : pred;
  predEl.className = 'pred-cls ' + (COLORS[pred] || 'cls-none');
  set('pred-name', d.prediction ? (NAMES[pred] || '—') : 'No data yet');

  const r = d.reason;
  if      (r === 'warmup')         set('pred-reason', `Warming up — ${d.n_samples} / 31 samples`);
  else if (r === 'pms_invalid')    set('pred-reason', 'PMS frame corrupted — row skipped');
  else if (r === 'low_confidence') set('pred-reason', 'Below confidence gate (0.40)');
  else                             set('pred-reason', '');

  // Stats
  set('s1',       fmt(d.s1_proba));
  set('conf',     fmt(d.confidence));
  set('n-samples', d.n_samples ?? 0);
  set('warmup',   d.warmed_up ? '✓ Ready' : '⏳ Warming');
  $('s1').className   = d.s1_proba   != null ? 'stat-v' : 'stat-v muted';
  $('conf').className = d.confidence != null ? 'stat-v' : 'stat-v muted';

  // Stage 2 bars
  if (d.s2_proba) {
    for (const c of ['A','C','K']) {
      const pct = (d.s2_proba[c] * 100).toFixed(1);
      $('bar-'+c).style.width = pct + '%';
      set('pct-'+c, pct + '%');
    }
  } else {
    for (const c of ['A','C','K']) { $('bar-'+c).style.width='0%'; set('pct-'+c,'—'); }
  }

  // Sensors
  const s = d.sensors;
  if (s) {
    for (const k of ['mq135','mq4','mq5','mq6','mq7','mq8'])
      set('v-'+k, s[k] ?? '—');
    set('v-temp', s.temperature_c != null ? Number(s.temperature_c).toFixed(1) : '—');
    set('v-hum',  s.humidity_pct  != null ? Number(s.humidity_pct).toFixed(1)  : '—');
    set('v-pm1',  s.pm1_0_atm ?? '—');
    set('v-pm25', s.pm2_5_atm ?? '—');
    set('v-pm10', s.pm10_atm  ?? '—');
    const ok = s.pms_valid === 1 || s.pms_valid === true;
    $('pms-dot').style.background = ok ? '#4ade80' : '#f87171';
    set('pms-txt', ok ? 'PMS frame valid' : 'PMS frame invalid');
  }

  set('tick', 'Last polled: ' + new Date().toLocaleTimeString());
}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>"""
