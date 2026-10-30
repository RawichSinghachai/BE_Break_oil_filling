"""
แสดงกราฟแนวโน้มของทุกคอลัมน์ (numeric) โดยเลือกดูได้ตาม lh_pgnum
ใช้ไฟล์ output ของ filter_data.py (เฉพาะ LH, ไม่มี null, recorded_at เป็น UTC+7)
"""
import math
import pandas as pd
import matplotlib.pyplot as plt

# ==================== ตั้งค่า ====================
INPUT = "data_filtered.csv"   # ไฟล์ที่ได้จาก filter_data.py
PGNUM = 20                    # <-- เลือก pgnum ที่ต้องการดู
COLOR_BY_BN = True            # True = แยกสีตามรถแต่ละคัน (lh_bn), False = เส้นเดียวรวมทุกคัน
KIND = "line"                 # "line" หรือ "scatter"
GRID_COLS = 3                 # จำนวนคอลัมน์ของ subplot grid
SAVE_FIG = True               # บันทึกรูปเป็น PNG
# =================================================

# คอลัมน์ที่เป็น key/identifier ไม่นำมา plot เป็นค่า
ID_COLS = {"lh_pgnum", "lh_bn", "recorded_at"}

# ---------- โหลดข้อมูล ----------
df = pd.read_csv(INPUT)

pgnums = sorted(df["lh_pgnum"].unique())
if PGNUM not in pgnums:
    raise SystemExit(f"ไม่พบ pgnum={PGNUM}. pgnum ที่มี: {[int(p) for p in pgnums]}")

sub = df[df["lh_pgnum"] == PGNUM].copy()
sub["recorded_at"] = pd.to_datetime(sub["recorded_at"], format="ISO8601")
sub = sub.sort_values("recorded_at").reset_index(drop=True)

# ---------- เลือกเฉพาะคอลัมน์ตัวเลขที่จะ plot ----------
plot_cols = [c for c in sub.columns
             if c not in ID_COLS and pd.api.types.is_numeric_dtype(sub[c])]
if not plot_cols:
    raise SystemExit("ไม่พบคอลัมน์ตัวเลขสำหรับ plot")

n_bn = sub["lh_bn"].nunique()
print(f"pgnum={PGNUM} : {len(sub):,} แถว, {n_bn} คัน, plot {len(plot_cols)} คอลัมน์")

# ---------- สร้าง subplot grid ----------
n = len(plot_cols)
rows = math.ceil(n / GRID_COLS)
fig, axes = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 5, rows * 3), squeeze=False)
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    ax = axes[i]
    if COLOR_BY_BN:
        for bn, g in sub.groupby("lh_bn"):
            if KIND == "scatter":
                ax.scatter(g["recorded_at"], g[col], s=6, label=str(bn))
            else:
                ax.plot(g["recorded_at"], g[col], lw=0.8, label=str(bn))
    else:
        if KIND == "scatter":
            ax.scatter(sub["recorded_at"], sub[col], s=6)
        else:
            ax.plot(sub["recorded_at"], sub[col], lw=0.8)
    ax.set_title(col, fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3)

# ซ่อนแกนที่เหลือเมื่อ grid ไม่เต็ม
for j in range(n, len(axes)):
    axes[j].set_visible(False)

# legend รวม (เฉพาะตอนแยกสีและจำนวนคันไม่มาก จะได้ไม่รก)
if COLOR_BY_BN and n_bn <= 15:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="lh_bn", loc="upper right", fontsize=7, ncol=2)

fig.suptitle(f"Data trends - pgnum {PGNUM}", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])

if SAVE_FIG:
    out = f"trend_pgnum{PGNUM}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"บันทึกกราฟเป็น {out}")

plt.show()