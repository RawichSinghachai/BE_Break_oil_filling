import numpy as np
import pandas as pd
import torch
import joblib
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import FEATURES, NORM_COLS, CSV_PATH, BATCH_SIZE, PGNUM, PGNUM_COL, BN_COL, SCALER_FILE

CSV_PATH = CSV_PATH
BATCH_SIZE = BATCH_SIZE

# คอลัมน์ที่ป้อนเข้าโมเดล (F = 8)
FEATURES = FEATURES
# คอลัมน์ที่ต้องการ normalize (ที่ไม่ใส่ = คงค่าเดิม)
NORM_COLS = NORM_COLS
idx = [FEATURES.index(c) for c in NORM_COLS]

# 1) โหลด + กรอง pgnum=20 + เรียงตาม id (เวลา)
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[df[PGNUM_COL] == PGNUM].sort_values("id").reset_index(drop=True)

# 2) แต่ละคัน (lh_bn) -> tensor (T_i, F) แล้ว pad เป็น (N, T, F)
#    groupby(sort=False) เก็บลำดับตามที่เจอก่อน-หลัง จึงยังเรียงตามเวลา
seqs = [torch.tensor(g[FEATURES].to_numpy(np.float32))
        for _, g in df.groupby(BN_COL, sort=False)]
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
joblib.dump(scaler, SCALER_FILE)                              # เก็บไว้ใช้ตอน inference

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
    print(name, X[s].shape)   # train (290, T, 8) / val (62, T, 8) / test (63, T, 8)


# แปลงกลับเป็นหน่วยเดิม (ใช้ตอนต้องการดูค่าจริงของ output)
def inverse(X_scaled, mask, scaler):
    out = X_scaled.clone()
    real = out[mask].numpy()
    real[:, idx] = scaler.inverse_transform(real[:, idx])
    out[mask] = torch.from_numpy(real)
    return out

# ส่งเลข bn  
groups = list(df.groupby("lh_bn", sort=False))
seqs = [torch.tensor(g[FEATURES].to_numpy(np.float32)) for _, g in groups]
bns  = np.array([k for k, _ in groups])  