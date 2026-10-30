# prepare_data.py

สคริปต์เตรียมข้อมูลสำหรับ LSTM Autoencoder (anomaly detection ของเครื่องเติมน้ำมันเบรก)

สิ่งที่สคริปต์ทำ:

1. โหลด CSV แล้วกรองเฉพาะ `lh_pgnum == 20` และเรียงตาม `id`
2. แยกข้อมูลเป็นรายคัน (`lh_bn`) แล้ว pad ให้ความยาวเท่ากัน ได้ tensor `(N, T, F)`
3. แบ่ง train / val / test = 70 / 15 / 15 ตามลำดับเวลา (ไม่สุ่ม)
4. Normalize ด้วย `StandardScaler` (fit จาก train เท่านั้น) และบันทึกเป็น `scaler.pkl`
5. สร้าง `DataLoader` สำหรับ train / val / test
6. มีฟังก์ชัน `inverse` สำหรับแปลงค่ากลับเป็นหน่วยเดิม

---

## 1. สิ่งที่ต้องมี

```bash
pip install numpy pandas torch scikit-learn joblib
```

## 2. ข้อมูลนำเข้า (CSV)

ไฟล์ตาม `CSV_PATH` (ค่าเริ่มต้น `../filter/data_filtered.csv`) ต้องมีคอลัมน์

| คอลัมน์ | ใช้ทำอะไร |
|---|---|
| `id` | เรียงลำดับตามเวลา |
| `lh_pgnum` | กรองเฉพาะค่า 20 |
| `lh_bn` | รหัสคัน ใช้แยกข้อมูลเป็น 1 sequence ต่อ 1 คัน |
| คอลัมน์ใน `FEATURES` | ฟีเจอร์ที่ป้อนเข้าโมเดล |

## 3. ค่าที่ปรับได้ (ส่วนบนของไฟล์)

| ตัวแปร | ความหมาย |
|---|---|
| `CSV_PATH` | ที่อยู่ไฟล์ CSV |
| `BATCH_SIZE` | จำนวนคันต่อ batch (ลดลงถ้า memory ไม่พอ) |
| `FEATURES` | คอลัมน์ที่ป้อนเข้าโมเดล (ปัจจุบัน F = 8) |
| `NORM_COLS` | คอลัมน์ที่ต้อง normalize ที่ไม่ใส่จะคงค่าเดิม (ปัจจุบันคือทุกคอลัมน์ยกเว้น `lh_stepnum`) |
| `test_size` ใน `train_test_split` | สัดส่วนการแบ่ง (0.30 แล้วแบ่งครึ่งอีกรอบ = 70/15/15) |

ข้อควรระวัง: ชื่อใน `FEATURES` ต้องไม่ซ้ำกัน เพราะ `idx` ใช้ `FEATURES.index(...)` ซึ่งคืนตำแหน่งแรกเสมอ คอลัมน์ที่ซ้ำจะไม่ถูก normalize

## 4. วิธีรัน

**รันเป็นสคริปต์**

```bash
python prepare_data.py
```

จะพิมพ์ shape ของแต่ละชุด เช่น

```
train torch.Size([290, 5752, 8])
val   torch.Size([62, 5752, 8])
test  torch.Size([63, 5752, 8])
```

และสร้างไฟล์ `scaler.pkl` ในโฟลเดอร์ปัจจุบัน (ตัวเลขจริงขึ้นกับข้อมูล)

**ใช้ใน Jupyter Notebook**

```python
%run prepare_data.py
```

ตัวแปรทั้งหมด (`X`, `mask`, `loaders`, `scaler`, `inverse` ฯลฯ) จะเข้ามาใน notebook

**import จากไฟล์อื่น**

```python
from prepare_data import X, mask, sl, loaders, scaler, inverse
```

สคริปต์ไม่มี `if __name__ == "__main__"` การ import จึงรันทั้งไฟล์ (โหลด CSV และ fit scaler ใหม่) ทุกครั้ง ถ้าไม่ต้องการ ให้บันทึกผลไว้แล้วโหลดทีหลัง (ดูหัวข้อ 6)

## 5. ตัวแปรที่ได้

| ตัวแปร | shape / ชนิด | ความหมาย |
|---|---|---|
| `X` | `(N, T, F)` | ข้อมูลทุกคัน normalize แล้ว padding เป็น 0 |
| `mask` | `(N, T)` bool | `True` = แถวจริง, `False` = padding |
| `lengths` | `(N,)` | จำนวนแถวจริงของแต่ละคัน |
| `sl` | dict ของ index tensor | index ของคันในแต่ละชุด: `sl["train"]`, `sl["val"]`, `sl["test"]` |
| `scaler` | `StandardScaler` | scaler ที่ fit จาก train (บันทึกที่ `scaler.pkl`) |
| `idx` | list | ตำแหน่งคอลัมน์ใน `FEATURES` ที่ถูก normalize |
| `loaders` | dict ของ `DataLoader` | `loaders["train"]`, `loaders["val"]`, `loaders["test"]` |

- `N` = จำนวนคัน, `T` = ความยาวของคันที่ยาวที่สุด, `F` = จำนวนฟีเจอร์
- แต่ละ batch จาก `loaders` ได้ `(x, m)` โดย `x` shape `(B, T, F)` และ `m` shape `(B, T)`

## 6. วิธีเรียกใช้ข้อมูล

### 6.1 ใน training loop

```python
for x, m in loaders["train"]:           # x: (B, T, F), m: (B, T)
    x_recon = model(x)
    loss = ((x_recon - x) ** 2)[m].mean()   # คิด loss เฉพาะแถวจริง
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

ใช้ `loaders["val"]` ตอนวัด validation loss (ครอบด้วย `model.eval()` และ `torch.no_grad()`)

### 6.2 ดึงทั้งชุดเป็น tensor เดียว (วิเคราะห์ผล)

```python
X_test = X[sl["test"]]          # (N_test, T, F)
m_test = mask[sl["test"]]       # (N_test, T)

model.eval()
with torch.no_grad():
    X_recon = model(X_test)

# anomaly score ต่อคัน = MSE เฉลี่ยเฉพาะแถวจริง
err = ((X_recon - X_test) ** 2).mean(-1)              # (N_test, T)
score = (err * m_test).sum(1) / m_test.sum(1)         # (N_test,)
```

การคิด score ในสเกลที่ normalize แล้วทำให้ทุกฟีเจอร์มีน้ำหนักเท่ากัน ไม่จำเป็นต้องแปลงกลับ

### 6.3 แปลงกลับเป็นหน่วยเดิม

```python
X_recon_real = inverse(X_recon, m_test, scaler)       # (N_test, T, F)
```

- แปลงเฉพาะแถวจริง ตำแหน่ง padding ไม่ถูกแตะ จึงควรดูเฉพาะ `X_recon_real[m_test]`
- `lh_stepnum` ไม่อยู่ใน `NORM_COLS` ค่าจึงเท่าเดิมทั้งขาไปและขากลับ
- ถ้าต้องการเทียบกับค่าดิบ ให้เก็บ `X_raw = X.clone()` ไว้ก่อนขั้นตอน normalize

### 6.4 ดึงทีละคัน

```python
x1 = X_test[0:1]                # (1, T, F) ใช้ 0:1 เพื่อคง dimension ของ batch
m1 = m_test[0:1]
```

### 6.5 Real-time ทีละ cycle (ข้อมูลใหม่ที่ยังไม่ normalize)

```python
import joblib, numpy as np, torch

scaler = joblib.load("scaler.pkl")

# df_cycle = ข้อมูลของ cycle ปัจจุบัน (กรอง lh_pgnum == 20 และเรียงตาม id แล้ว)
x = torch.tensor(df_cycle[FEATURES].to_numpy(np.float32))        # (T_i, F)
x[:, idx] = torch.from_numpy(scaler.transform(x[:, idx].numpy()))
x = x.unsqueeze(0)                                               # (1, T_i, F)
m = torch.ones(1, x.shape[1], dtype=torch.bool)                  # ไม่มี padding

with torch.no_grad():
    x_recon = model(x)
score = ((x_recon - x) ** 2)[m].mean().item()
```

ห้าม fit scaler ใหม่ตอน real-time ให้ใช้ `scaler.pkl` ที่ได้จากตอน train เสมอ

### 6.6 บันทึก / โหลด dataset ข้ามไฟล์

เพิ่มท้าย `prepare_data.py`

```python
torch.save({"X": X, "mask": mask, "lengths": lengths, "sl": sl}, "dataset.pt")
```

ตอนเทรนในไฟล์อื่น

```python
d = torch.load("dataset.pt")
X, mask, sl = d["X"], d["mask"], d["sl"]
scaler = joblib.load("scaler.pkl")
```

## 7. หลักการสำคัญ

- **Mask**: คันแต่ละคันมีความยาวไม่เท่ากัน จึง pad ด้วย 0 ให้ครบ `T` และใช้ `mask` แยกแถวจริงออกจาก padding ทั้งตอน fit scaler, ตอน inverse และตอนคิด loss ถ้าไม่ใช้ mask โมเดลจะเรียนรู้ที่จะ reconstruct padding และ loss จะดูดีเกินจริง
- **Scaler fit จาก train เท่านั้น**: เพื่อไม่ให้สถิติของข้อมูลช่วงหลัง (val/test) รั่วเข้ามาตอน train
- **แบ่งตามเวลา**: `shuffle=False` ทำให้ train เป็นช่วงแรกสุด และ test เป็นช่วงหลังสุด ลำดับของคันอ้างอิงจากการเรียงตาม `id` และ `groupby(sort=False)`
- **Padding ถึงความยาวสูงสุดของทั้งชุด**: ถ้า memory หรือเวลาเทรนเป็นปัญหา ให้ลด `BATCH_SIZE` หรือ trim padding ต่อ batch ให้สั้นลงตามคันที่ยาวที่สุดใน batch นั้น

## 8. ปัญหาที่พบบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `KeyError` ชื่อคอลัมน์ | ชื่อใน `FEATURES` ไม่ตรงกับ header ใน CSV |
| คอลัมน์หนึ่งไม่ถูก normalize | ชื่อซ้ำใน `FEATURES` / `NORM_COLS` |
| out of memory | ลด `BATCH_SIZE` หรือ trim padding ต่อ batch |
| anomaly score เพี้ยนตอน real-time | ใช้ scaler ที่ fit ใหม่ แทนที่จะโหลด `scaler.pkl` เดิม |
| ค่า `inverse` ตรง padding ผิดปกติ | เป็นเรื่องปกติ ให้เลือกดูเฉพาะแถวจริงด้วย `[mask]` |