"""
ระบุ lh_pgnum แล้วแสดงรายการ lh_bn (รถแต่ละคัน) ที่มีอยู่ใน pgnum นั้น
พร้อมจำนวนค่าอ่าน (rows) และช่วงเวลา recorded_at ของแต่ละคัน

ใช้ไฟล์ที่ผ่านการ cleaning แล้ว (output ของ filter_data.py) ซึ่งมีเฉพาะฝั่ง LH,
ไม่มี null, ผ่านเงื่อนไข continue/master_on/machine_fault และ recorded_at เป็น UTC+7 อยู่แล้ว
จึงไม่ต้องตัดคอลัมน์ rh / แปลงเวลา / dropna ซ้ำ
"""
import pandas as pd

# ==================== ตั้งค่า ====================
INPUT = "data_filtered.csv"   # ไฟล์ที่ได้จาก filter_data.py
PGNUM = 14                        # <-- ระบุ lh_pgnum ที่นี่
SAVE_CSV = True                   # True = บันทึกรายการเป็นไฟล์ CSV ด้วย
# =================================================

# ---------- โหลดเฉพาะคอลัมน์ที่ใช้ ----------
lh = pd.read_csv(INPUT, usecols=["lh_pgnum", "lh_bn", "recorded_at"])

# ตรวจสอบ pgnum
pgnums = sorted(lh["lh_pgnum"].unique())
if PGNUM not in pgnums:
    raise SystemExit(f"ไม่พบ pgnum={PGNUM}. pgnum ที่มี: {[int(p) for p in pgnums]}")

sub = lh[lh["lh_pgnum"] == PGNUM].copy()

# recorded_at ใน CSV เป็น string ที่มี offset +07:00 อยู่แล้ว -> แปลงเป็น datetime เฉพาะแถวของ pgnum นี้
sub["recorded_at"] = pd.to_datetime(sub["recorded_at"], format="ISO8601")

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