"""
วิเคราะห์ข้อมูลฝั่ง LH จากไฟล์ machine data
ขั้นตอน:
  1) แยกเฉพาะฝั่ง LH (ตัดคอลัมน์ที่ขึ้นต้นด้วย 'rh' ออก)
  2) เรียงข้อมูลตาม lh_pgnum -> lh_bn -> lh_stepnum
  3) plot กราฟคอลัมน์ที่เลือก (YCOL) ของรถในแต่ละ lh_pgnum
     พร้อมแรเงาพื้นหลังตาม stage (lh_stepnum) "จริง" ที่แต่ละตำแหน่ง

ตั้งค่า YCOL เพื่อเลือกคอลัมน์ที่จะแสดงบนแกน y
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ==================== ตั้งค่า ====================
INPUT = "data.csv"                 # เปลี่ยนเป็น path ไฟล์ของคุณ
YCOL  = "lh_vacuumgun"       # <-- เลือกคอลัมน์ที่จะแสดงบนแกน y
# lh_fillingpressure
# lh_fillingvolume
# lh_vacuumgun
# =================================================

# ---------- 1) โหลดข้อมูล + แยกฝั่ง LH ----------
df = pd.read_csv(INPUT, low_memory=False)

# เก็บเฉพาะคอลัมน์ที่ "ไม่" ขึ้นต้นด้วย rh  -> ได้ฝั่ง LH + คอลัมน์กลางที่ใช้ร่วมกัน
lh_cols = [c for c in df.columns if not c.startswith("rh")]
lh = df[lh_cols].copy()

# ตรวจสอบ YCOL (คอลัมน์แกน y) -> ต้องเป็นคอลัมน์ตัวเลขฝั่ง LH (ไม่ใช่คีย์/ป้ายกำกับ)
not_y = {"lh_pgnum", "lh_bn", "lh_stepnum", "lh_asn", "id"}
y_choices = [c for c in lh.columns
             if c not in not_y and pd.api.types.is_numeric_dtype(lh[c])]
if YCOL not in lh.columns:
    print(f"ไม่พบคอลัมน์ YCOL='{YCOL}'. คอลัมน์ที่เลือกได้: {y_choices}")
    sys.exit(0)

# ---------- 2) เรียงข้อมูล ----------
# เรียงตาม lh_pgnum -> lh_bn -> lh_stepnum (stage การทำงาน 0,1,2)
# เพิ่ม id ท้ายสุด เพื่อให้ลำดับการอ่านภายใน stage เดียวกันคงที่ (ตามเวลา)
sort_keys = ["lh_pgnum", "lh_bn", "lh_stepnum"]
if "id" in lh.columns:
    sort_keys.append("id")
lh_sorted = lh.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

# บันทึกผลลัพธ์ที่แยก+เรียงแล้ว
lh_sorted.to_csv("lh_sorted.csv", index=False)
print(f"บันทึก lh_sorted.csv : {lh_sorted.shape[0]:,} แถว x {lh_sorted.shape[1]} คอลัมน์")

# ---------- 3) plot YCOL + แรเงาพื้นหลังตาม stage จริง ----------
# หนึ่ง subplot ต่อหนึ่ง lh_pgnum ; แต่ละเส้น (สีเข้ม) คือรถหนึ่งคัน
# แกน x = ลำดับการอ่านค่าภายในรถ (หลัง sort = stage0 -> stage1 -> stage2 ตามเวลา)
# พื้นหลัง = สีจางตาม lh_stepnum "จริง": ที่ตำแหน่ง k ดูว่ารถส่วนใหญ่อยู่ stage ไหน
#            (modal stepnum) แล้วลงสีคอลัมน์นั้นตาม stage ที่พบมากสุด

STAGE_COLORS = {0.0: "#4C78A8", 1.0: "#F58518", 2.0: "#54A24B"}
STAGE_LABEL  = {0.0: "stage 0", 1.0: "stage 1", 2.0: "stage 2"}

pgnums = sorted(lh_sorted["lh_pgnum"].dropna().unique())
n = len(pgnums)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                         squeeze=False)

for i, pg in enumerate(pgnums):
    ax = axes[i // ncols][i % ncols]
    sub = lh_sorted[lh_sorted["lh_pgnum"] == pg].dropna(subset=["lh_bn"]).copy()
    n_cars = sub["lh_bn"].nunique()

    # ช่วงแกน x: ตัดหางรถ outlier ที่ยาวผิดปกติ ให้เห็นรอบการเติมจริง
    lens = sub.groupby("lh_bn").size()
    xmax = lens.quantile(0.95)

    # ---- พื้นหลังตาม stage จริง ----
    sub["pos"] = sub.groupby("lh_bn").cumcount()
    modal = (sub[sub["pos"] <= xmax]
             .dropna(subset=["lh_stepnum"])
             .groupby("pos")["lh_stepnum"]
             .agg(lambda s: s.value_counts().idxmax()))
    if len(modal):
        pos = modal.index.to_numpy()
        stg = modal.to_numpy()
        rs = pos[0]
        for j in range(1, len(pos) + 1):
            if j == len(pos) or stg[j] != stg[j - 1]:
                st = stg[j - 1]
                if st in STAGE_COLORS:
                    ax.axvspan(rs - 0.5, pos[j - 1] + 0.5,
                               color=STAGE_COLORS[st], alpha=0.15, lw=0)
                if j < len(pos):
                    rs = pos[j]

    # ---- เส้นค่า YCOL ของรถแต่ละคัน (สีเข้มกลาง โปร่งแสง) ----
    for _, car in sub.groupby("lh_bn"):
        y = car[YCOL].to_numpy()
        ax.plot(range(len(y)), y, color="#2a2a2a", alpha=0.18, linewidth=0.6)

    ax.set_xlim(0, xmax)
    ax.set_title(f"lh_pgnum = {int(pg)}  ({n_cars} cars)", fontsize=10)
    ax.set_xlabel("reading index (within car)")
    ax.set_ylabel(YCOL)
    ax.grid(True, alpha=0.3)

# ซ่อน subplot ที่เหลือว่าง
for j in range(n, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

# legend รวมสำหรับ stage (สีพื้นหลัง = stepnum จริง)
from matplotlib.patches import Patch
handles = [Patch(facecolor=STAGE_COLORS[s], alpha=0.35, label=STAGE_LABEL[s])
           for s in [0.0, 1.0, 2.0]]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
           bbox_to_anchor=(0.99, 1.0))

fig.suptitle(f"{YCOL} per car, grouped by lh_pgnum "
             "(background = actual lh_stepnum stage)", fontsize=14, y=1.002)
fig.tight_layout()
OUT = f"{YCOL}_by_pgnum.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"บันทึกกราฟ {OUT}")