"""
plot กราฟ LH + RH รวมในกราฟเดียวกัน จาก new/data_filtered_all.csv (output ของ main_all.py)

ไฟล์ data_filtered_all.csv ผ่านขั้นตอนเหล่านี้มาแล้ว จึงไม่ต้องทำซ้ำในสคริปต์นี้:
  - ตัดคอลัมน์ l0-l7 ; เอา rh ต่อท้าย lh (คอลัมน์ชื่อไม่มี prefix + คอลัมน์ side)
  - ลบ row ที่เป็น null
  - เลือก continue=True & master_on=True & machine_fault=False
  - เรียง pgnum -> bn -> side -> stepnum
  - เก็บเฉพาะรอบ (side, pgnum, bn) ที่มี step 0-1-2 ครบ
  - recorded_at เป็น UTC+7

lh และ rh ถูกมองเป็นฝั่งเดียวกัน: วาดเส้นสีเดียวกันใน subplot ของ pgnum เดียวกัน
ตั้งค่า YCOL เพื่อเลือกคอลัมน์ที่จะแสดงบนแกน y
"""

import sys
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==================== ตั้งค่า ====================
HERE  = Path(__file__).resolve().parent          # โฟลเดอร์ new/
INPUT = HERE / "data_filtered_all.csv"           # ไฟล์ที่ได้จาก main_all.py
YCOL  = "vacuumgun"                              # <-- เลือกคอลัมน์ที่จะแสดงบนแกน y
# fillingpressure
# fillingvolume
# vacuumgun
# vacuumpump
# =================================================

CAR = ["side", "bn"]                      # รถหนึ่งคัน = (side, bn)
KEYS = ["pgnum", "bn", "side", "stepnum"]  # ลำดับการเรียงของไฟล์
NOT_Y = {"side", "pgnum", "bn", "stepnum", "asn", "id",
         "continue", "master_on", "machine_fault", "recorded_at"}

# ---------- 1) โหลดข้อมูลที่ clean แล้ว ----------
header = pd.read_csv(INPUT, nrows=0).columns
y_choices = [c for c in header if c not in NOT_Y]
if YCOL not in header or YCOL in NOT_Y:
    print(f"ไม่พบคอลัมน์ YCOL='{YCOL}'. คอลัมน์ที่เลือกได้: {y_choices}")
    sys.exit(0)

# อ่านเฉพาะคอลัมน์ที่ใช้ plot (เร็วและกินหน่วยความจำน้อยลง)
data = pd.read_csv(INPUT, usecols=sorted(set(KEYS + [YCOL])))

# ไฟล์ต้องเรียงมาแล้ว เพราะ "reading index" ใช้ลำดับแถวในไฟล์
# (ถ้าไม่เรียง ให้รัน main_all.py ใหม่ ไม่ต้อง sort ซ้ำที่นี่)
if not data[KEYS].equals(data[KEYS].sort_values(KEYS, kind="mergesort")):
    sys.exit(f"{INPUT} ไม่ได้เรียงตาม {KEYS} - รัน main_all.py ใหม่ก่อน")
print(f"โหลด {INPUT.name} : {data.shape[0]:,} แถว "
      f"(lh {int((data['side'] == 'lh').sum()):,} / rh {int((data['side'] == 'rh').sum()):,})")

# ---------- 2) plot YCOL + แรเงาพื้นหลังตาม stage จริง ----------
# หนึ่ง subplot ต่อหนึ่ง pgnum ; แต่ละเส้น (สีเข้ม) คือรถหนึ่งคัน (lh และ rh รวมกัน)
# แกน x = ลำดับการอ่านค่าภายในรถ (stage0 -> stage1 -> stage2)
# พื้นหลัง = สีจางตาม stepnum "จริง": ที่ตำแหน่ง k ใช้ stage ที่พบมากสุด (modal)

STAGE_COLORS = {0.0: "#4C78A8", 1.0: "#F58518", 2.0: "#54A24B"}
STAGE_LABEL  = {0.0: "stage 0", 1.0: "stage 1", 2.0: "stage 2"}

pgnums = sorted(data["pgnum"].unique())
n = len(pgnums)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                         squeeze=False)

for i, pg in enumerate(pgnums):
    ax = axes[i // ncols][i % ncols]
    sub = data[data["pgnum"] == pg].copy()
    n_cars = sub.groupby(CAR).ngroups

    # ช่วงแกน x: ตัดหางรถ outlier ที่ยาวผิดปกติ ให้เห็นรอบการเติมจริง
    lens = sub.groupby(CAR).size()
    xmax = lens.quantile(0.95)

    # ---- พื้นหลังตาม stage จริง ----
    sub["pos"] = sub.groupby(CAR).cumcount()
    modal = (sub[sub["pos"] <= xmax]
             .groupby("pos")["stepnum"]
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

    # ---- เส้นค่า YCOL ของรถแต่ละคัน (lh/rh สีเดียวกัน) ----
    for _, car in sub.groupby(CAR):
        y = car[YCOL].to_numpy()
        ax.plot(range(len(y)), y, color="#2a2a2a", alpha=0.18, linewidth=0.6)

    ax.set_xlim(0, xmax)
    ax.set_title(f"pgnum = {int(pg)}  ({n_cars} cars)", fontsize=10)
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

fig.suptitle(f"{YCOL} per car (lh + rh), grouped by pgnum "
             "(background = actual stepnum stage)", fontsize=14, y=1.002)
fig.tight_layout()
OUT = HERE / f"{YCOL}_by_pgnum_all.png"
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"บันทึกกราฟ {OUT}")
