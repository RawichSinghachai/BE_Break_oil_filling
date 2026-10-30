"""
พล็อต lh_fillingpressure ของรถ 1 คัน พร้อมแรเงาพื้นหลังตาม stage (lh_stepnum) จริง

วิธีใช้: แก้ค่าในส่วน "ตั้งค่า" ด้านล่าง
  PGNUM        : เลือกโปรแกรมการผลิต (lh_pgnum)
  BN           : เลือกรถ (lh_bn) ที่ต้องการ
                 - ใส่เลข bn ที่ต้องการ เช่น BN = 26747
                 - หรือปล่อยเป็น None  -> เลือกคันที่ความยาวใกล้ค่ากลางให้อัตโนมัติ
                 - ถ้าใส่ bn ที่ไม่มีใน pgnum -> print แจ้งว่าไม่มี แล้วจบการทำงาน
  USE_DATETIME : เลือกแกน x
                 - False -> ใช้ลำดับการอ่าน range(len(car))
                 - True  -> ใช้เวลา recorded_at ตามที่อยู่ใน CSV จริง (ไม่แปลง timezone)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ==================== ตั้งค่า ====================
INPUT = "data.csv"              # path ไฟล์ข้อมูล
PGNUM = 20                      # <-- เลือก lh_pgnum ที่นี่
BN    = 26802                   # <-- เลือก lh_bn ที่นี่ (ใส่เลข หรือ None = เลือกอัตโนมัติ)
YCOL  = "lh_vacuumgun"    # <-- เลือกคอลัมน์ที่จะแสดงบนแกน y
USE_DATETIME = False            # <-- True = แกน x เป็น recorded_at (เวลาไทย) ; False = index
# =================================================

# ---------- โหลด + แยกฝั่ง LH + เรียง ----------
df = pd.read_csv(INPUT, low_memory=False)
lh = df[[c for c in df.columns if not c.startswith("rh")]].copy()
lh = lh.sort_values(["lh_pgnum", "lh_bn", "lh_stepnum", "id"]).reset_index(drop=True)

# ตรวจสอบ YCOL (คอลัมน์แกน y)
# คอลัมน์ที่เลือกได้ = คอลัมน์ตัวเลขฝั่ง LH (ไม่รวมคีย์/ป้ายกำกับ)
not_y = {"lh_pgnum", "lh_bn", "lh_stepnum", "lh_asn", "id"}
y_choices = [c for c in lh.columns
             if c not in not_y and pd.api.types.is_numeric_dtype(lh[c])]
if YCOL not in lh.columns:
    print(f"ไม่พบคอลัมน์ YCOL='{YCOL}'. คอลัมน์ที่เลือกได้: {y_choices}")
    sys.exit(0)

# ตรวจสอบ PGNUM
pgnums = sorted(lh["lh_pgnum"].dropna().unique())
if PGNUM not in pgnums:
    print(f"ไม่พบ pgnum={PGNUM}. pgnum ที่มี: {[int(p) for p in pgnums]}")
    sys.exit(0)

sub = lh[lh["lh_pgnum"] == PGNUM].dropna(subset=["lh_bn"])
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

car = sub[sub["lh_bn"] == target_bn].reset_index(drop=True)
y = car[YCOL].to_numpy()
steps = car["lh_stepnum"].to_numpy()

# ---------- เลือกแกน x ----------
# ทุกจุดวางห่างเท่ากันเสมอ (ใช้ลำดับเป็นตำแหน่ง) ; เมื่อ USE_DATETIME=True จะใส่
# ป้ายกำกับแกนเป็นเวลาไทยจริงของแต่ละจุด (ไม่ยืด/หดตามช่องว่างเวลาจริง)
x = np.arange(len(car))
if USE_DATETIME:
    t_thai = pd.to_datetime(car["recorded_at"], errors="coerce").dt.tz_convert("Asia/Bangkok")
    xlabel = "recorded_at (Thai time, Asia/Bangkok) - equal spacing"
else:
    xlabel = "reading index (within car)"
print(f"pgnum={PGNUM}  bn={int(target_bn)}  readings={len(car)}  "
      f"x-axis={'recorded_at (เวลาไทย, equal spacing)' if USE_DATETIME else 'index'}")

# ขอบของแต่ละจุด (กึ่งกลางระหว่างจุดข้างเคียง) ใช้วางแถบ stage ให้ชิดกันไม่มีช่องว่าง
# x เป็นลำดับตัวเลขเสมอ -> ขอบคือ ±0.5 ของแต่ละจุด
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

# เส้นแรงดัน
ax.plot(x, y, color="#2a2a2a", linewidth=1.3)
ax.scatter(x, y, color="#2a2a2a", s=8, zorder=3)

ax.set_title(f"{YCOL} — pgnum {PGNUM}, car bn={int(target_bn)} "
             f"(background = actual lh_stepnum)", fontsize=12)
ax.set_xlabel(xlabel)
ax.set_ylabel(YCOL)
ax.grid(True, alpha=0.3)
if USE_DATETIME:
    # จุดห่างเท่ากัน แต่ป้ายกำกับเป็นเวลาไทยจริงของจุดนั้น (เลือกแสดงไม่เกิน ~12 ป้าย)
    ntick = min(12, n)
    idx = np.unique(np.linspace(0, n - 1, ntick).round().astype(int))
    ax.set_xticks(x[idx])
    ax.set_xticklabels([t_thai.iloc[i].strftime("%H:%M:%S") for i in idx],
                       rotation=45, ha="right")
ax.legend(handles=[Patch(facecolor=STAGE_COLORS[s], alpha=0.35, label=STAGE_LABEL[s])
                   for s in [0.0, 1.0, 2.0]], loc="lower right", frameon=True)

fig.tight_layout()
OUT = (f"{YCOL}_pgnum{PGNUM}_bn{int(target_bn)}"
       f"_{'time' if USE_DATETIME else 'index'}.png")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"บันทึก {OUT}")