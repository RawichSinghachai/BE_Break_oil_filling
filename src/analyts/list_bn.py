"""
ระบุ lh_pgnum แล้วแสดงรายการ lh_bn (รถแต่ละคัน) ที่มีอยู่ใน pgnum นั้น
พร้อมจำนวนค่าอ่าน (rows) และช่วงเวลา recorded_at ของแต่ละคัน
"""
import pandas as pd

# ==================== ตั้งค่า ====================
INPUT = "data.csv"          # path ไฟล์ข้อมูล
PGNUM = 20                  # <-- ระบุ lh_pgnum ที่นี่
TZ    = "Asia/Bangkok"      # เขตเวลาที่ใช้แสดง (ข้อมูลดิบเป็น UTC)
SAVE_CSV = True             # True = บันทึกรายการเป็นไฟล์ CSV ด้วย
# =================================================

# ---------- โหลด + แยกฝั่ง LH ----------
df = pd.read_csv(INPUT, low_memory=False)
lh = df[[c for c in df.columns if not c.startswith("rh")]].copy()
lh["recorded_at"] = pd.to_datetime(lh["recorded_at"], errors="coerce").dt.tz_convert(TZ)

# ตรวจสอบ pgnum
pgnums = sorted(lh["lh_pgnum"].dropna().unique())
if PGNUM not in pgnums:
    raise SystemExit(f"ไม่พบ pgnum={PGNUM}. pgnum ที่มี: {[int(p) for p in pgnums]}")

sub = lh[lh["lh_pgnum"] == PGNUM].dropna(subset=["lh_bn"])

# ---------- สรุปรายการ bn ----------
summary = (sub.groupby("lh_bn")
              .agg(rows=("lh_bn", "size"),
                   first_seen=("recorded_at", "min"),
                   last_seen=("recorded_at", "max"))
              .reset_index()
              .sort_values("first_seen")       # เรียงตามเวลาที่เริ่มผลิต
              .reset_index(drop=True))
summary["lh_bn"] = summary["lh_bn"].astype("Int64")

bn_list = summary["lh_bn"].tolist()
print(f"pgnum={PGNUM} : มี lh_bn ทั้งหมด {len(bn_list)} คัน, รวม {len(sub):,} แถว\n")
print("รายการ lh_bn ทั้งหมด:")
print(bn_list)
print("\nรายละเอียดต่อคัน (เรียงตามเวลาเริ่มผลิต):")
# แสดงทุกแถวแบบไม่ตัด
with pd.option_context("display.max_rows", None, "display.width", 120):
    print(summary.to_string(index=False))

if SAVE_CSV:
    out = f"bn_list_pgnum{PGNUM}.csv"
    summary.to_csv(out, index=False)
    print(f"\nบันทึกรายการเป็น {out}")