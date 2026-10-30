"""
train.py — เทรน + ตั้ง threshold เป็น % ความต่าง + ประเมิน
ได้ไฟล์ (ต่อท้ายด้วย _pg<NN>_<side> เช่น _pg20_lh): lstm_ae_*.pt, thresholds_*.pkl,
loss_curve_*.png, recon_*_*.png, pct_scores_*.png
"""
import math
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_data import X, mask, sl, loaders, scaler, inverse, bns
# from model2 import LSTMAutoencoder, masked_mse, anomaly_score
from model3 import LSTMAutoencoder, masked_mse, anomaly_score
from config import (FEATURES, CHANNELS, MAX_LEN, STEP_IDX, TAG,
                    MODEL_FILE, THRESHOLDS_FILE, EPOCHS, LR, PATIENCE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- ตั้ง % threshold เองที่นี่ (เกิน % นี้ = abnormal) ----------
PCT_THRESHOLDS = {"pressure": 8.0, "volume": 8.0, "vacuumgun": 20.0}
Q = 99                                                    # ใช้เฉพาะโหมด auto (percentile ของ % บน val)


# ---------- helper: % ความต่าง (หารด้วยค่าเฉลี่ยของรอบ) ----------
def percent_diff(input_seq, recon_seq):
    a = np.asarray(input_seq, dtype=float)
    b = np.asarray(recon_seq, dtype=float)
    mae = np.abs(b - a).mean()
    pct = mae / (np.abs(a).mean() + 1e-6) * 100
    return pct, mae


# ---------- helper: % ต่อคัน ของทั้ง split ----------
def cycle_percents(model, split, ch):
    """คืน array ของ % ความต่าง 1 ค่าต่อคัน สำหรับ channel ch"""
    model.eval()
    s = sl[split]
    x_raw = inverse(X[s], mask[s], scaler)                 # หน่วยจริง
    with torch.no_grad():
        recon = model(X[s].to(DEVICE), mask[s].to(DEVICE)).cpu()
    r_raw = inverse(recon, mask[s], scaler)
    pcts = []
    for i in range(len(s)):
        L = int(mask[s][i].sum())
        pcts.append(percent_diff(x_raw[i, :L, ch], r_raw[i, :L, ch])[0])
    return np.array(pcts)


# ---------- ขั้น 4: เทรน + early stopping ----------
def train():
    model = LSTMAutoencoder(n_feat=len(FEATURES),
                            step_idx=STEP_IDX,
                            max_len=MAX_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, wait, best_state, hist = float("inf"), 0, None, []

    for ep in range(1, EPOCHS + 1):
        model.train(); tot = n = 0
        for x, m in loaders["train"]:
            x, m = x.to(DEVICE), m.to(DEVICE)
            opt.zero_grad()
            loss = masked_mse(model(x, m), x, m)
            loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        tr = tot / n

        model.eval(); vt = vn = 0
        with torch.no_grad():
            for x, m in loaders["val"]:
                x, m = x.to(DEVICE), m.to(DEVICE)
                vt += masked_mse(model(x, m), x, m).item() * len(x); vn += len(x)
        va = vt / vn
        hist.append((tr, va))
        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:3d}  train {tr:.4f}  val {va:.4f}")

        if va < best - 1e-5:
            best, wait, best_state = va, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"early stop @ epoch {ep} (best val {best:.4f})"); break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_FILE)
    return model, hist


# ---------- กราฟ loss ----------
def plot_loss(hist):
    hist = np.array(hist)
    best_ep = int(hist[:, 1].argmin()) + 1
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(hist) + 1), hist[:, 0], label="train loss")
    plt.plot(range(1, len(hist) + 1), hist[:, 1], label="val loss")
    plt.axvline(best_ep, color="gray", ls=":", label=f"best val @ {best_ep}")
    plt.xlabel("epoch"); plt.ylabel("masked MSE loss")
    plt.title("Training / Validation loss"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"loss_curve_{TAG}.png", dpi=110)
    print(f"saved loss_curve_{TAG}.png  (best val epoch = {best_ep})")


# ---------- ขั้น 5: threshold เป็น % ----------
def set_thresholds(model=None, custom=PCT_THRESHOLDS, auto=False):
    """
    ตั้ง % threshold ต่อ feature แล้วเซฟ thresholds.pkl
      custom : dict กำหนดค่า % เอง เช่น {"pressure":5, "volume":5, "vacuumgun":15}
      auto   : True = คิด p99 ของ % จาก val แทน (ต้องส่ง model), ไม่สน custom
    """
    if auto:
        if model is None:
            raise ValueError("auto=True ต้องส่ง model มาด้วย")
        thr = {name: float(np.percentile(cycle_percents(model, "val", ch), Q))
               for name, ch in CHANNELS.items()}
    else:
        thr = {name: float(custom[name]) for name in CHANNELS}   # ใช้ค่าที่กำหนดเอง
    joblib.dump(thr, THRESHOLDS_FILE)
    print("thresholds (% diff):", {k: round(v, 2) for k, v in thr.items()})
    return thr


# ---------- ขั้น 6: ประเมิน (ตัดสินด้วย %) ----------
def evaluate(model, thr):
    for name, ch in CHANNELS.items():
        p = cycle_percents(model, "test", ch)
        flagged = int((p > thr[name]).sum())
        print(f"{name:9s} thr={thr[name]:.2f}%  flagged {flagged}/{len(p)} test cars")


# ---------- ภาพ input vs recon + % ใน title ----------
def plot_recon(model, channel_name="volume", n=9, ncols=3):
    model.eval()
    s = sl["test"][:n]
    x, m = X[s].to(DEVICE), mask[s].to(DEVICE)
    with torch.no_grad():
        recon = model(x, m).cpu()
    x_raw = inverse(X[s], mask[s], scaler)
    r_raw = inverse(recon, mask[s], scaler)
    ch = CHANNELS[channel_name]

    nrows = math.ceil(n / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    ax = ax.ravel()
    s = sl["test"][:n]

    for i in range(n):
        L = int(mask[s][i].sum())
        inp = x_raw[i, :L, ch]; rec = r_raw[i, :L, ch]
        pct, _ = percent_diff(inp, rec)
        car_bn = int(bns[s[i]])
        ax[i].plot(inp, label="input"); ax[i].plot(rec, "--", label="recon")
        ax[i].set_title(f"bn {car_bn} — {channel_name}  (diff {pct:.1f}%)")
        ax[i].legend(fontsize=8)
    for j in range(n, len(ax)):
        ax[j].axis("off")
    plt.tight_layout(); plt.savefig(f"recon_{channel_name}_{TAG}.png", dpi=110)
    print(f"saved recon_{channel_name}_{TAG}.png")


# ---------- % diff per car, by feature + threshold line ----------
def plot_pct_scores(model, thr, split="test"):
    names = list(CHANNELS.keys())
    fig, ax = plt.subplots(len(names), 1, figsize=(11, 3 * len(names)), sharex=True)
    for a, name in zip(ax, names):
        p = cycle_percents(model, split, CHANNELS[name]); t = thr[name]; flag = p > t
        a.scatter(np.where(~flag)[0], p[~flag], s=28, color="#2563eb", label="normal")
        a.scatter(np.where(flag)[0],  p[flag],  s=55, color="#dc2626", zorder=3, label="exceed threshold")
        a.axhline(t, color="#b45309", ls="--", lw=1.6, label=f"threshold={t:.2f}%")
        a.set_ylabel(f"{name}\n% diff"); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=.25)
    ax[-1].set_xlabel("car index (test)")
    fig.suptitle("Percent difference per car by feature", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"pct_scores_{TAG}.png", dpi=110)
    print(f"saved pct_scores_{TAG}.png")


if __name__ == "__main__":
    model, hist = train()
    plot_loss(hist)
    thr = set_thresholds(custom=PCT_THRESHOLDS)     # กำหนด % เอง (หรือ set_thresholds(model, auto=True))
    evaluate(model, thr)
    plot_recon(model, "volume",   n=20)
    plot_recon(model, "pressure", n=20)
    plot_recon(model, "vacuumgun", n=20)
    plot_pct_scores(model, thr)