from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent                              # โฟลเดอร์ ml/new/
CSV_PATH = HERE / "../../filter/new/data_filtered_all.csv"          # lh+rh (output ของ main_all.py)
SCALER_PATH = HERE / "scaler.pkl"
BATCH_SIZE = 16
PGNUM = 20

# คอลัมน์ในไฟล์ใหม่ไม่มี prefix lh_/rh_ แล้ว (มี column "side" บอกฝั่ง)
# คอลัมน์ที่ป้อนเข้าโมเดล (F = 8)
FEATURES = [
    "stepnum",
    "fillingpressure", "fillingvolume", "vacuumgun",
    "vacuumpump", "pumpvacuumcurrent", "pumpfillingcurrent",
    "pumpsupplycurrent",
]
# คอลัมน์ที่ต้องการ normalize (ที่ไม่ใส่ = คงค่าเดิม)
NORM_COLS = [
    "fillingpressure", "fillingvolume", "vacuumgun",
    "vacuumpump", "pumpvacuumcurrent", "pumpfillingcurrent",
    "pumpsupplycurrent",
]
idx = [FEATURES.index(c) for c in NORM_COLS]

# รถหนึ่งคัน = (side, bn)  เพราะ bn อาจซ้ำกันคนละฝั่ง
CAR = ["side", "bn"]

# 1) โหลด + กรอง pgnum=20 (lh และ rh รวมกัน) + เรียงตาม id (เวลา)
#    lh/rh ของแถวเดิมมี id เท่ากัน -> ใช้ mergesort (stable) ให้ lh มาก่อน rh คงที่
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[df["pgnum"] == PGNUM].sort_values(["id", "side"], kind="mergesort").reset_index(drop=True)
print("rows:", df.shape[0], "| cars:", df.groupby(CAR).ngroups,
      "| lh:", df[df["side"] == "lh"].groupby("bn").ngroups,
      "rh:", df[df["side"] == "rh"].groupby("bn").ngroups)

# 2) แต่ละคัน (side, bn) -> tensor (T_i, F) แล้ว pad เป็น (N, T, F)
#    groupby(sort=False) เก็บลำดับตามที่เจอก่อน-หลัง จึงยังเรียงตามเวลา
groups = list(df.groupby(CAR, sort=False))
seqs = [torch.tensor(g[FEATURES].to_numpy(np.float32)) for _, g in groups]
lengths = torch.tensor([len(s) for s in seqs])
X = pad_sequence(seqs, batch_first=True)                       # (N, T, F)
mask = torch.arange(X.shape[1])[None, :] < lengths[:, None]    # (N, T) True = แถวจริง

# 3) แบ่ง train / val / test ตามลำดับเวลา 70/15/15 (ไม่สุ่ม)
ids = np.arange(X.shape[0])
tr, tmp = train_test_split(ids, test_size=0.30, shuffle=False)   # 70 / 30
va, te = train_test_split(tmp, test_size=0.50, shuffle=False)    # 15 / 15
sl = {
    "train": torch.from_numpy(tr),
    "val":   torch.from_numpy(va),
    "test":  torch.from_numpy(te),
}

# 4) fit scaler จากแถวจริงของ train เท่านั้น แล้ว transform ทั้งชุด
real_train = X[sl["train"]][mask[sl["train"]]].numpy()         # (แถวจริงของ train, F)
scaler = StandardScaler().fit(real_train[:, idx])
joblib.dump(scaler, SCALER_PATH)                               # เก็บไว้ใช้ตอน inference

real = X[mask].numpy()                                         # แถวจริงทั้งหมด
real[:, idx] = scaler.transform(real[:, idx])
X[mask] = torch.from_numpy(real)                               # padding ยังเป็น 0

# 5) Dataset / DataLoader (แต่ละ batch ได้ (x, mask))
loaders = {}
for name, s in sl.items():
    ds = TensorDataset(X[s], mask[s])
    loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE,
                               shuffle=(name == "train"))      # shuffle เฉพาะ train

for name, s in sl.items():
    print(name, X[s].shape)   # (จำนวนคัน, T, 8)


# แปลงกลับเป็นหน่วยเดิม (ใช้ตอนต้องการดูค่าจริงของ output)
def inverse(X_scaled, mask, scaler):
    out = X_scaled.clone()
    real = out[mask].numpy()
    real[:, idx] = scaler.inverse_transform(real[:, idx])
    out[mask] = torch.from_numpy(real)
    return out

# เลข bn และฝั่งของแต่ละคัน (ลำดับตรงกับแถวใน X)
bns   = np.array([k[1] for k, _ in groups])
sides = np.array([k[0] for k, _ in groups])
