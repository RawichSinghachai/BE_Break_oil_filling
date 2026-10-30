"""
config.py — ตั้งค่าที่ใช้ร่วมทุกไฟล์ไว้ที่เดียว
เปลี่ยน pgnum / side (lh,rh) / feature ที่นี่ที่เดียว ไม่ต้องไล่แก้หลายไฟล์
"""

# รุ่น + ข้างที่กำลังทำงานด้วย (เปลี่ยน 2 บรรทัดนี้แล้วรัน train.py / predict.py ใหม่)
PGNUM = 20
SIDE  = "lh"          # "lh" หรือ "rh"  <-- สลับข้างที่นี่ที่เดียว

CSV_PATH   = "../filter/data_filtered.csv"
BATCH_SIZE = 16
MAX_LEN    = 120                                   # ความยาวรอบสูงสุด (model3.py + predict.py)

# feature ที่ป้อนเข้าโมเดล — คอลัมน์ที่ขึ้นกับข้างเติม {SIDE}_ ให้อัตโนมัติ
# ส่วน pump ร่วม (pumpfillingcurrent/pumpsupplycurrent) ไม่มี prefix ใช้ค่าเดียวทั้งสองข้าง
FEATURES = [
    f"{SIDE}_stepnum",
    f"{SIDE}_fillingpressure", f"{SIDE}_fillingvolume", f"{SIDE}_vacuumgun",
    f"{SIDE}_vacuumpump", f"{SIDE}_pumpvacuumcurrent", "pumpfillingcurrent",
    "pumpsupplycurrent",
]

# คอลัมน์ที่ต้อง normalize (ที่ไม่อยู่ในนี้ = คงค่าเดิม)
NORM_COLS = [
    f"{SIDE}_fillingpressure", f"{SIDE}_fillingvolume", f"{SIDE}_vacuumgun",
    f"{SIDE}_vacuumpump", f"{SIDE}_pumpvacuumcurrent", "pumpfillingcurrent",
    "pumpsupplycurrent",
]

# channel ที่ "ตรวจ" (index อ้างตามตำแหน่งใน FEATURES)
CHANNELS = {
    "pressure":  FEATURES.index(f"{SIDE}_fillingpressure"),
    "volume":    FEATURES.index(f"{SIDE}_fillingvolume"),
    "vacuumgun": FEATURES.index(f"{SIDE}_vacuumgun"),
}

# คอลัมน์/ตำแหน่งที่ไฟล์อื่นต้องใช้ แต่ไม่ต้องรู้ว่าข้างไหน (derive จาก SIDE ให้แล้ว)
STEP_IDX  = FEATURES.index(f"{SIDE}_stepnum")       # ตำแหน่ง stepnum
PGNUM_COL = f"{SIDE}_pgnum"                          # คอลัมน์กรอง pgnum
BN_COL    = f"{SIDE}_bn"                             # คอลัมน์ group รายคัน
SENTINEL_COLS = (f"{SIDE}_vacuumgun", f"{SIDE}_vacuumpump")   # คอลัมน์ที่ 9999 = missing

# ต่อท้ายชื่อไฟล์ output ทุกไฟล์ เช่น scaler_pg20_lh.pkl, lstm_ae_pg20_rh.pt
# -> เก็บได้ครบทุก pgnum/ข้าง ไม่ทับกัน
TAG = f"pg{PGNUM}_{SIDE}"

# ชื่อไฟล์ output (ผูกกับ TAG อยู่แล้ว เปลี่ยน PGNUM/SIDE ก็เลื่อนตามเอง)
SCALER_FILE     = f"scaler_{TAG}.pkl"       # prepare_data เซฟ / predict โหลด
MODEL_FILE      = f"lstm_ae_{TAG}.pt"        # train เซฟ / predict โหลด
THRESHOLDS_FILE = f"thresholds_{TAG}.pkl"    # train เซฟ / predict โหลด

# ---------- พารามิเตอร์เทรน (train.py) ----------
EPOCHS   = 3000
LR       = 1e-3
PATIENCE = 10