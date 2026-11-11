from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent                      # โฟลเดอร์ filter/new/
df = pd.read_csv(HERE / "../../analyts/data.csv", low_memory=False)
print("raw:", df.shape)

# แปลง recorded_at จาก UTC เป็น UTC+7 (Asia/Bangkok)
df["recorded_at"] = pd.to_datetime(df["recorded_at"], format="ISO8601", utc=True).dt.tz_convert("Asia/Bangkok")

# 1) ตัด column l0-l7 (ใน data มี l0-l6)
df = df.drop(columns=[c for c in df.columns if c in {f"l{i}" for i in range(8)}])
print("after drop cols:", df.shape)

# 2) แยก lh / rh แล้วเอา rh มาต่อท้าย lh
#    column ที่ไม่ขึ้นต้นด้วย lh_/rh_ เป็นข้อมูลร่วม -> ใส่ซ้ำทั้งสองฝั่ง
#    ตัด prefix ออกให้ชื่อ column ตรงกัน และเพิ่ม column "side" บอกว่าเป็นฝั่งไหน
shared_cols = [c for c in df.columns if not c.startswith(("lh_", "rh_"))]
parts = []
for side in ("lh", "rh"):
    side_cols = [c for c in df.columns if c.startswith(f"{side}_")]
    part = df[shared_cols + side_cols].rename(columns=lambda c: c[len(side) + 1:] if c.startswith(f"{side}_") else c)
    part.insert(0, "side", side)
    print(f"  {side}:", part.shape)
    parts.append(part)
df = pd.concat(parts, ignore_index=True)
print("after concat lh+rh:", df.shape)

# 3) ลบ row ที่มี null
df = df.dropna()
print("after dropna:", df.shape)

# 4) เลือกเงื่อนไข
df = df[(df["continue"] == True) & (df["master_on"] == True) & (df["machine_fault"] == False)]
print("after condition:", df.shape)

# 5) จำกัดเพดานค่าในคอลัมน์ vacuum: ค่าที่มากกว่า 300 ให้ใส่เป็น 300
#    (ค่า spike เช่น 2600 / 9999 ถูกตัดเหลือ 300 แทนการเปลี่ยนเป็น 0 เพื่อไม่ให้เส้นดิ่งลงเป็นรูปตัว V)
CLIP_COLS = ["vacuumpump", "vacuumgun"]
THRESH = 300
for c in CLIP_COLS:
    if c in df.columns:
        n = int((df[c] > THRESH).sum())
        print(f"  {c}: ค่า > {THRESH} ใส่เป็น {THRESH} ({n} จุด)")
        df[c] = df[c].clip(upper=THRESH)

# 6) เรียงข้อมูล pgnum, bn, stepnum (side ใช้แยก group เผื่อ bn ซ้ำกันคนละฝั่ง)
GROUP = ["side", "pgnum", "bn"]
df = df.sort_values(["pgnum", "bn", "side", "stepnum"]).reset_index(drop=True)

# 7) เก็บเฉพาะ group (side, pgnum, bn) ที่เริ่มจาก step 0 -> 1 -> 2 (มีครบทั้ง 3 step)
steps = df.groupby(GROUP)["stepnum"].agg(lambda s: set(s))
valid = steps[steps.map(lambda s: {0, 1, 2} <= s)].index
mask = pd.MultiIndex.from_frame(df[GROUP]).isin(valid)
df = df[mask].reset_index(drop=True)
print("after step 0-1-2:", df.shape, "| groups:", len(valid), "of", len(steps))

# 8) ตัด group (side, pgnum, bn) ที่ยาวผิดปกติ ด้วย IQR high (ตัวคูณ 6)
#    คำนวณขอบบนแยกในแต่ละ pgnum (รวมทั้ง lh และ rh) เพราะแต่ละโปรแกรมมีความยาวปกติไม่เท่ากัน
K = 6
length = df.groupby(GROUP).size()
g = length.groupby(level="pgnum")
q1 = g.transform(lambda s: s.quantile(0.25))
q3 = g.transform(lambda s: s.quantile(0.75))
high = q3 + K * (q3 - q1)

removed = length[length > high]
summary = pd.DataFrame({
    "groups": length.groupby(level="pgnum").size(),
    "removed": removed.groupby(level="pgnum").size(),
    "iqr_high": high.groupby(level="pgnum").first(),
}).fillna({"removed": 0}).astype({"removed": int})
print(summary)

keep = length[length <= high].index
mask = pd.MultiIndex.from_frame(df[GROUP]).isin(keep)
df = df[mask].reset_index(drop=True)
print(f"after IQR high (K={K}):", df.shape, "| groups:", len(keep), "of", len(length))

df.to_csv(HERE / "data_filtered_all.csv", index=False)
