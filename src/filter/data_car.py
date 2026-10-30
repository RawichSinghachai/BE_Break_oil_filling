"""
พล็อต YCOL ของรถ 1 คัน พร้อมแรเงาพื้นหลังตาม stage (lh_stepnum) จริง

ใช้ไฟล์ที่ผ่านการ cleaning แล้ว (output ของ filter_data.py) ซึ่งมีเฉพาะฝั่ง LH,
ไม่มี null, เรียง lh_pgnum -> lh_bn -> lh_stepnum -> id แล้ว และ recorded_at เป็น UTC+7 อยู่แล้ว
จึงไม่ต้องตัดคอลัมน์ rh / sort / แปลงเวลาซ้ำ

วิธีใช้: แก้ค่าในส่วน "ตั้งค่า" ด้านล่าง
  PGNUM        : เลือกโปรแกรมการผลิต (lh_pgnum)
  BN           : เลือกรถ (lh_bn) - ใส่เลข bn หรือ None = เลือกคันที่ความยาวใกล้ค่ากลางให้อัตโนมัติ
                 ถ้าใส่ bn ที่ไม่มีใน pgnum -> print แจ้งว่าไม่มี แล้วจบการทำงาน
  USE_DATETIME : False -> ป้ายแกน x เป็นลำดับการอ่าน
                 True  -> ป้ายแกน x เป็นเวลา recorded_at (UTC+7) ; จุดยังวางห่างเท่ากัน
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==================== ตั้งค่า ====================
INPUT = "data_filtered.csv"   # ไฟล์ที่ได้จาก filter_data.py
PGNUM = 14                        # <-- เลือก lh_pgnum ที่นี่
BN    = 27605                     # <-- เลือก lh_bn ที่นี่ (ใส่เลข หรือ None = เลือกอัตโนมัติ)
YCOL  = "lh_vacuumpump"            # <-- เลือกคอลัมน์ที่จะแสดงบนแกน y
USE_DATETIME = False              # <-- True = ป้ายแกน x เป็น recorded_at ; False = index
# =================================================

NOT_Y = {"lh_pgnum", "lh_bn", "lh_stepnum", "lh_asn", "id",
         "continue", "master_on", "machine_fault", "recorded_at"}

# ---------- โหลดข้อมูลที่ clean แล้ว ----------
header = pd.read_csv(INPUT, nrows=0).columns
y_choices = [c for c in header if c not in NOT_Y]
if YCOL not in header or YCOL in NOT_Y:
    print(f"ไม่พบคอลัมน์ YCOL='{YCOL}'. คอลัมน์ที่เลือกได้: {y_choices}")
    sys.exit(0)

cols = ["lh_pgnum", "lh_bn", "lh_stepnum", YCOL] + (["recorded_at"] if USE_DATETIME else [])
lh = pd.read_csv(INPUT, usecols=cols)

# ตรวจสอบ PGNUM
pgnums = sorted(lh["lh_pgnum"].unique())
if PGNUM not in pgnums:
    print(f"ไม่พบ pgnum={PGNUM}. pgnum ที่มี: {[int(p) for p in pgnums]}")
    sys.exit(0)

sub = lh[lh["lh_pgnum"] == PGNUM]
lens = sub.groupby("lh_bn").size()

# เลือกรถ
if BN is None:
    target_bn = (lens - lens.median()).abs().idxmin()
    print(f"BN=None -> เลือกคันปกติอัตโนมัติ: bn={int(target_bn)}")
else:
    if BN not in lens.index:
        print(f"ไม่พบ bn={BN} ใน pgnum={PGNUM} - ไม่มีข้อมูลให้แสดง")
        sys.exit(0)
    target_bn = BN

# ไฟล์เรียงมาแล้ว -> ลำดับแถวของรถคันนี้คือ stage0 -> 1 -> 2 ตามเวลา
car = sub[sub["lh_bn"] == target_bn].reset_index(drop=True)
y = car[YCOL].to_numpy()
steps = car["lh_stepnum"].to_numpy()

# ---------- เลือกแกน x ----------
# ทุกจุดวางห่างเท่ากันเสมอ (ใช้ลำดับเป็นตำแหน่ง) ; เมื่อ USE_DATETIME=True จะใส่
# ป้ายกำกับแกนเป็นเวลาจริง (UTC+7) ของแต่ละจุด (ไม่ยืด/หดตามช่องว่างเวลาจริง)
x = np.arange(len(car))
if USE_DATETIME:
    # recorded_at ใน CSV มี offset +07:00 อยู่แล้ว ; ISO8601 รองรับแถวที่ไม่มีเศษวินาที
    t_thai = pd.to_datetime(car["recorded_at"], format="ISO8601")
    xlabel = "recorded_at (UTC+7) - equal spacing"
else:
    xlabel = "reading index (within car)"
print(f"pgnum={PGNUM}  bn={int(target_bn)}  readings={len(car)}  "
      f"x-axis={'recorded_at (UTC+7, equal spacing)' if USE_DATETIME else 'index'}")

# ขอบของแต่ละจุด (±0.5) ใช้วางแถบ stage ให้ชิดกันไม่มีช่องว่าง
n = len(x)
edges = np.concatenate([[x[0] - 0.5], x[:-1] + 0.5, [x[-1] + 0.5]]) if n >= 2 \
    else np.array([x[0] - 0.5, x[0] + 0.5])

# ---------- plot ----------
STAGE_COLORS = {0.0: "#4C78A8", 1.0: "#F58518", 2.0: "#54A24B"}
STAGE_LABEL  = {0.0: "stage 0", 1.0: "stage 1", 2.0: "stage 2"}

fig, ax = plt.subplots(figsize=(11, 5))

# แรเงาพื้นหลังตาม stage จริงของรถคันนี้ (รวมช่วงที่ stage เดียวกันติดกัน)
rs = 0
for k in range(1, n + 1):
    if k == n or steps[k] != steps[k - 1]:
        st = steps[k - 1]
        if st in STAGE_COLORS:
            ax.axvspan(edges[rs], edges[k], color=STAGE_COLORS[st], alpha=0.15, lw=0)
        rs = k

ax.plot(x, y, color="#2a2a2a", linewidth=1.3)
ax.scatter(x, y, color="#2a2a2a", s=8, zorder=3)

ax.set_title(f"{YCOL} — pgnum {PGNUM}, car bn={int(target_bn)} "
             f"(background = actual lh_stepnum)", fontsize=12)
ax.set_xlabel(xlabel)
ax.set_ylabel(YCOL)
ax.grid(True, alpha=0.3)
if USE_DATETIME:
    # ป้ายกำกับเป็นเวลาจริงของจุดนั้น (แสดงไม่เกิน ~12 ป้าย)
    ntick = min(12, n)
    idx = np.unique(np.linspace(0, n - 1, ntick).round().astype(int))
    ax.set_xticks(x[idx])
    ax.set_xticklabels([t_thai.iloc[i].strftime("%H:%M:%S") for i in idx],
                       rotation=45, ha="right")
ax.legend(handles=[Patch(facecolor=STAGE_COLORS[s], alpha=0.35, label=STAGE_LABEL[s])
                   for s in STAGE_COLORS], loc="lower right", frameon=True)

fig.tight_layout()
OUT = (f"{YCOL}_pgnum{PGNUM}_bn{int(target_bn)}"
       f"_{'time' if USE_DATETIME else 'index'}.png")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"บันทึก {OUT}")