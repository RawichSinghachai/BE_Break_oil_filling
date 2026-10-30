"""
plot กราฟฝั่ง LH จากไฟล์ที่ผ่านการ cleaning แล้ว (output ของ filter_data.py)

ไฟล์ data_filtered.csv ผ่านขั้นตอนเหล่านี้มาแล้ว จึงไม่ต้องทำซ้ำในสคริปต์นี้:
  - ตัดคอลัมน์ rh* และ l0-l7
  - ลบ row ที่เป็น null
  - เลือก continue=True & master_on=True & machine_fault=False
  - เรียง lh_pgnum -> lh_bn -> lh_stepnum
  - เก็บเฉพาะรอบ (pgnum, bn) ที่มี step 0-1-2 ครบ
  - recorded_at เป็น UTC+7

ตั้งค่า YCOL เพื่อเลือกคอลัมน์ที่จะแสดงบนแกน y
"""

import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==================== ตั้งค่า ====================
INPUT = "data_filtered.csv"   # ไฟล์ที่ได้จาก filter_data.py
YCOL  = "lh_vacuumgun"            # <-- เลือกคอลัมน์ที่จะแสดงบนแกน y
# lh_fillingpressure
# lh_fillingvolume
# lh_vacuumgun
# =================================================

KEYS = ["lh_pgnum", "lh_bn", "lh_stepnum"]
NOT_Y = {"lh_pgnum", "lh_bn", "lh_stepnum", "lh_asn", "id",
         "continue", "master_on", "machine_fault", "recorded_at"}

# ---------- 1) โหลดข้อมูลที่ clean แล้ว ----------
header = pd.read_csv(INPUT, nrows=0).columns
y_choices = [c for c in header if c not in NOT_Y]
if YCOL not in header or YCOL in NOT_Y:
    print(f"ไม่พบคอลัมน์ YCOL='{YCOL}'. คอลัมน์ที่เลือกได้: {y_choices}")
    sys.exit(0)

# อ่านเฉพาะคอลัมน์ที่ใช้ plot (เร็วและกินหน่วยความจำน้อยลง)
lh_sorted = pd.read_csv(INPUT, usecols=KEYS + [YCOL])

# ไฟล์ต้องเรียงมาแล้ว เพราะ "reading index" ใช้ลำดับแถวในไฟล์
# (ถ้าไม่เรียง ให้รัน filter_data.py ใหม่ ไม่ต้อง sort ซ้ำที่นี่)
if not lh_sorted[KEYS].equals(lh_sorted[KEYS].sort_values(KEYS, kind="mergesort")):
    sys.exit(f"{INPUT} ไม่ได้เรียงตาม {KEYS} - รัน filter_data.py ใหม่ก่อน")
print(f"โหลด {INPUT} : {lh_sorted.shape[0]:,} แถว")

# ---------- 2) plot YCOL + แรเงาพื้นหลังตาม stage จริง ----------
# หนึ่ง subplot ต่อหนึ่ง lh_pgnum ; แต่ละเส้น (สีเข้ม) คือรถหนึ่งคัน
# แกน x = ลำดับการอ่านค่าภายในรถ (stage0 -> stage1 -> stage2)
# พื้นหลัง = สีจางตาม lh_stepnum "จริง": ที่ตำแหน่ง k ใช้ stage ที่พบมากสุด (modal)

STAGE_COLORS = {0.0: "#4C78A8", 1.0: "#F58518", 2.0: "#54A24B"}
STAGE_LABEL  = {0.0: "stage 0", 1.0: "stage 1", 2.0: "stage 2"}

pgnums = sorted(lh_sorted["lh_pgnum"].unique())
n = len(pgnums)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                         squeeze=False)

for i, pg in enumerate(pgnums):
    ax = axes[i // ncols][i % ncols]
    sub = lh_sorted[lh_sorted["lh_pgnum"] == pg].copy()
    n_cars = sub["lh_bn"].nunique()

    # ช่วงแกน x: ตัดหางรถ outlier ที่ยาวผิดปกติ ให้เห็นรอบการเติมจริง
    lens = sub.groupby("lh_bn").size()
    xmax = lens.quantile(0.95)

    # ---- พื้นหลังตาม stage จริง ----
    sub["pos"] = sub.groupby("lh_bn").cumcount()
    modal = (sub[sub["pos"] <= xmax]
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
handles = [Patch(facecolor=STAGE_COLORS[s], alpha=0.35, label=STAGE_LABEL[s])
           for s in STAGE_COLORS]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
           bbox_to_anchor=(0.99, 1.0))

fig.suptitle(f"{YCOL} per car, grouped by lh_pgnum "
             "(background = actual lh_stepnum stage)", fontsize=14, y=1.002)
fig.tight_layout()
OUT = f"{YCOL}_by_pgnum.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"บันทึกกราฟ {OUT}")