"""
predict.py — ใช้งานโมเดลกับ 1 รอบ (รถ 1 คัน) คืน dict เดียวพร้อมทำเป็น API (JSON ได้เลย)

output (run_prediction):
  1. recon      : time-series ที่ predict ของทุก feature (dict ของ list)
  2. percent    : % ความต่าง ต่อ feature ที่ตรวจ
  3. pass       : True/False ต่อ feature (True = ผ่านเกณฑ์/ปกติ) + overall_pass
  4. input      : time-series ข้อมูลจริงทุก feature ก่อน predict (สำหรับแสดงบนเว็บ)
"""
import numpy as np
import pandas as pd
import torch
import joblib
from model3 import LSTMAutoencoder
from config import (FEATURES, NORM_COLS, CHANNELS, MAX_LEN, PGNUM,
                    STEP_IDX, PGNUM_COL, BN_COL, SENTINEL_COLS,
                    SCALER_FILE, MODEL_FILE, THRESHOLDS_FILE)

# ---------- ต้องตรงกับตอนเทรน ----------
FEATURES = FEATURES
NORM_COLS = NORM_COLS
IDX = [FEATURES.index(c) for c in NORM_COLS]
CHANNELS = CHANNELS
MAX_LEN = MAX_LEN
ROUND = 3                                                   # ปัดทศนิยมใน payload
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- โหลดครั้งเดียว ----------
scaler = joblib.load(SCALER_FILE)
thresholds = joblib.load(THRESHOLDS_FILE)                  # % threshold ต่อ channel
model = LSTMAutoencoder(n_feat=len(FEATURES),
                        step_idx=STEP_IDX,
                        max_len=MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()


def percent_diff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mae = np.abs(b - a).mean()
    return float(mae / (np.abs(a).mean() + 1e-6) * 100), float(mae)


def _norm(arr):
    out = arr.copy(); out[:, IDX] = scaler.transform(out[:, IDX]); return out

def _inv(arr):
    out = arr.copy(); out[:, IDX] = scaler.inverse_transform(out[:, IDX]); return out


def preprocess_cycle(df_cycle):
    g = df_cycle.sort_values("id").copy()
    for c in SENTINEL_COLS:
        g.loc[g[c] == 9999, c] = np.nan
    g[FEATURES] = g[FEATURES].ffill().bfill()
    raw = g[FEATURES].to_numpy("float32")                  # [T,F] หน่วยจริง
    x = torch.from_numpy(_norm(raw.copy())).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=DEVICE)
    return x, mask, raw


def _series(arr_2d):
    """array [T,F] -> dict {feature: [values]} (list ของ float, JSON ได้)"""
    return {FEATURES[j]: [round(float(v), ROUND) for v in arr_2d[:, j]]
            for j in range(len(FEATURES))}


def run_prediction(df_cycle):
    """คืน dict เดียว พร้อมเป็น JSON response ของ API"""
    x, mask, raw_in = preprocess_cycle(df_cycle)
    with torch.no_grad():
        recon = model(x, mask).cpu().numpy()[0]            # [T,F] normalized
    raw_rec = _inv(recon)                                  # หน่วยจริง
    bn = int(df_cycle.sort_values("id")[BN_COL].iloc[-1])
    T = int(raw_in.shape[0])

    # 2+3: % และ pass เฉพาะ feature ที่ตรวจ
    features = {}
    for name, ch in CHANNELS.items():
        pct, mae = percent_diff(raw_in[:, ch], raw_rec[:, ch])
        thr = float(thresholds[name])
        features[name] = {
            "column": FEATURES[ch],
            "percent": round(pct, 2),
            "mae": round(mae, 2),
            "threshold": round(thr, 2),
            "pass": bool(pct <= thr),                      # True = ผ่านเกณฑ์ (ปกติ)
            "input_value": round(float(raw_in[:, ch].mean()), 2),
            "recon_value": round(float(raw_rec[:, ch].mean()), 2),
        }

    return {
        "bn": bn,
        "length": T,
        "overall_pass": all(f["pass"] for f in features.values()),   # 3
        "features": features,                                          # 2 + 3
        "input": _series(raw_in),                                     # 4: ทุก feature ก่อน predict
        "recon": _series(raw_rec),                                    # 1: time-series ที่ predict
    }


if __name__ == "__main__":
    import sys, json
    df = pd.read_csv(sys.argv[1], low_memory=False)
    df = df[df[PGNUM_COL] == PGNUM]
    last_bn = df.sort_values("id")[BN_COL].iloc[-1]
    cycle = df[df[BN_COL] == last_bn]

    out = run_prediction(cycle)
    # สรุปสั้น ๆ
    print(f"bn {out['bn']}  overall_pass = {out['overall_pass']}  (T={out['length']})")
    for name, f in out["features"].items():
        print(f"  {name:9s} {f['percent']:5.1f}% / thr {f['threshold']:.1f}%  "
              f"input~{f['input_value']} recon~{f['recon_value']}  pass={f['pass']}")
    # payload เต็ม (JSON) -> เขียนไฟล์หรือส่งผ่าน API ได้เลย
    with open(f"pred_{out['bn']}.json", "w") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"saved pred_{out['bn']}.json")