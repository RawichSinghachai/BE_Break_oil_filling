import pandas as pd

df = pd.read_csv("../analyts/data.csv", low_memory=False)
print("raw:", df.shape)

# แปลง recorded_at จาก UTC เป็น UTC+7 (Asia/Bangkok)
df["recorded_at"] = pd.to_datetime(df["recorded_at"], format="ISO8601", utc=True).dt.tz_convert("Asia/Bangkok")

# 1) ตัด column ที่ขึ้นต้นด้วย rh และ l0-l7 (ใน data มี l0-l6)
drop_cols = [c for c in df.columns if c.startswith("rh")]
drop_cols += [c for c in df.columns if c in {f"l{i}" for i in range(8)}]
df = df.drop(columns=drop_cols)
print("after drop cols:", df.shape)

# 2) ลบ row ที่มี null
df = df.dropna()
print("after dropna:", df.shape)

# 3) เลือกเงื่อนไข
df = df[(df["continue"] == True) & (df["master_on"] == True) & (df["machine_fault"] == False)]
print("after condition:", df.shape)

# 4) แทนค่า sentinel (ค่าพุ่งชนเพดาน 10000) ด้วย 0 ในคอลัมน์ vacuum
#    ค่าปกติต่ำกว่า threshold มาก ส่วน spike = 10000 ไม่ใช่ค่าจริง
SENTINEL_COLS = ["lh_vacuumpump", "lh_vacuumgun"]
THRESH = 300
for c in SENTINEL_COLS:
    if c in df.columns:
        n = int((df[c] >= THRESH).sum())
        print(f"  {c}: แทนค่า >= {THRESH} ด้วย 0 ({n} จุด)")
        df.loc[df[c] >= THRESH, c] = 0

# 5) เรียงข้อมูล pgnum, bn, stepnum
df = df.sort_values(["lh_pgnum", "lh_bn", "lh_stepnum"]).reset_index(drop=True)

# 6) เก็บเฉพาะ group (pgnum, bn) ที่เริ่มจาก step 0 -> 1 -> 2 (มีครบทั้ง 3 step)
steps = df.groupby(["lh_pgnum", "lh_bn"])["lh_stepnum"].agg(lambda s: set(s))
valid = steps[steps.map(lambda s: {0, 1, 2} <= s)].index
mask = pd.MultiIndex.from_frame(df[["lh_pgnum", "lh_bn"]]).isin(valid)
df = df[mask].reset_index(drop=True)
print("after step 0-1-2:", df.shape, "| groups:", len(valid), "of", len(steps))

# 7) ตัด group (pgnum, bn) ที่ยาวผิดปกติ ด้วย IQR high (ตัวคูณ 6)
#    คำนวณขอบบนแยกในแต่ละ pgnum เพราะแต่ละโปรแกรมมีความยาวปกติไม่เท่ากัน
K = 6
length = df.groupby(["lh_pgnum", "lh_bn"]).size()
g = length.groupby(level="lh_pgnum")
q1 = g.transform(lambda s: s.quantile(0.25))
q3 = g.transform(lambda s: s.quantile(0.75))
high = q3 + K * (q3 - q1)

removed = length[length > high]
summary = pd.DataFrame({
    "groups": length.groupby(level="lh_pgnum").size(),
    "removed": removed.groupby(level="lh_pgnum").size(),
    "iqr_high": high.groupby(level="lh_pgnum").first(),
}).fillna({"removed": 0}).astype({"removed": int})
print(summary)

keep = length[length <= high].index
mask = pd.MultiIndex.from_frame(df[["lh_pgnum", "lh_bn"]]).isin(keep)
df = df[mask].reset_index(drop=True)
print(f"after IQR high (K={K}):", df.shape, "| groups:", len(keep), "of", len(length))

df.to_csv("data_filtered.csv", index=False)