"""
cycle.py — ประกอบแถวรายวินาทีจากตาราง machine ให้เป็น "รถทีละคัน" (logic ล้วน ไม่แตะ DB / โมเดล)

กฎตัดรอบ เหมือน cycle.py ฝั่ง OPC (เครื่อง 1) : ตัดตรงจุดที่ step เปลี่ยนจาก "ไม่ใช่ 0 -> 0"
  คันเก่าปิด (ไม่รวม 0 ตัวนี้), 0 ตัวนี้เป็นแถวแรกของคันใหม่ ; เก็บทุกแถวรวม 0 ที่ติดกัน

ส่วนที่เพิ่ม/แก้จากของเดิม (ดูผลรีวิวจากข้อมูลจริงใน DB):
  1. เพดานความยาว max_len (ค่าเริ่มต้น 120 = MAX_LEN ของโมเดล): สายหยุดแล้ว step ค้างที่ 2 รอบจะไม่จบเอง
     (เจอรอบยาว 450-7000 แถว) -> ปล่อยรอบทันทีเมื่อครบเพดาน แล้วไม่เก็บแถวต่อจนกว่าจะเจอ step 0 ใหม่
  2. แถวที่ step เป็น None ไม่ทิ้ง: เก็บไว้ และถือว่า step เท่าเดิม
  3. ข้ามรอบแรกที่เริ่มกลางคัน: ยังไม่ปล่อยรอบใดจนกว่าจะเห็นจุดตัดครั้งแรก
  4. ตรวจก่อนปล่อย: ต้องเริ่มที่ step 0, ถึง step 2 และยาว >= min_len ไม่ผ่านก็ข้าม (+ log)
  5. bn / pgnum / asn อ่านจาก "แถวสุดท้าย" ของรอบ: แถวแรก (step 0) บางครั้ง (~7%) ยังเป็นค่าของรถคันก่อน
     แต่ยังเก็บแถวแรกไว้ในข้อมูล เพราะระบบเดิม/ตอนเทรนรวมมันอยู่แล้ว
"""
import logging
from dataclasses import dataclass, field

log = logging.getLogger("cycle")

MAX_CYCLE_LEN = 120     # เท่ากับ MAX_LEN ใน predict.py (รอบปกติยาวสุดที่เจอ 119)
MIN_CYCLE_LEN = 60      # รอบปกติสั้นสุดที่เจอ 80


def select_side_columns(row: dict, side: str) -> dict:
    """คัด 1 แถว ให้เหลือเฉพาะคอลัมน์ของข้างนี้ + คอลัมน์กลาง (ตัดข้างตรงข้ามทิ้ง)"""
    other = "rh_" if side.upper() == "LH" else "lh_"
    return {k: v for k, v in row.items() if not k.lower().startswith(other)}


def side_to_lh(row: dict, side: str) -> dict:
    """เตรียมแถวให้โมเดลอ่านได้: โมเดลเทรนด้วยคอลัมน์ lh_* ฝั่ง RH จึงต้องเปลี่ยนชื่อ rh_* -> lh_* (คอลัมน์กลางคงเดิม)
    ใช้โมเดลเดียวกันกับทั้ง LH และ RH ; ฝั่ง LH คืนแถวเดิม"""
    if side.upper() == "LH":
        return row
    return {("lh_" + k[3:] if k.lower().startswith("rh_") else k): v
            for k, v in row.items() if not k.lower().startswith("lh_")}


@dataclass
class Cycle:
    """รอบของรถ 1 คัน (ของข้างหนึ่ง) ที่ประกอบเสร็จแล้ว"""
    side: str                     # 'LH' / 'RH'
    bn: int | None                # จากแถวสุดท้าย
    pgnum: int | None             # จากแถวสุดท้าย
    asn: int | None               # จากแถวสุดท้าย
    rows: list[dict] = field(default_factory=list)   # เรียงตาม recorded_at, คัดคอลัมน์ข้างนี้แล้ว
    truncated: bool = False       # True = ถูกตัดที่เพดาน max_len (สายหยุดค้างที่ step 2)


class CycleBuffer:
    """สะสมแถวของข้างหนึ่ง (LH หรือ RH) แล้วคายรอบออกมาเมื่อจบคัน"""

    def __init__(self, side: str, max_len: int = MAX_CYCLE_LEN, min_len: int = MIN_CYCLE_LEN):
        self.side = side.upper()
        self.max_len = max_len
        self.min_len = min_len
        self.step_key = f"{self.side.lower()}_stepnum"
        self.rows: list[dict] = []
        self.prev_step = 0
        self.synced = False       # เห็นจุดตัดครั้งแรกแล้วหรือยัง (ก่อนหน้านั้นเป็นรอบที่เริ่มกลางคัน)
        self.capped = False       # รอบนี้ถูกปล่อยที่เพดานไปแล้ว -> ไม่เก็บแถวต่อจนกว่าจะเจอ step 0 ใหม่
        self.last_reject: str | None = None   # เหตุผลที่รอบล่าสุดถูกข้าม (ไว้ให้เทสต์/ดีบัก)

    def update(self, row: dict) -> Cycle | None:
        """ป้อน 1 แถว (dict คีย์ตัวพิมพ์เล็ก เช่น lh_stepnum, lh_bn, recorded_at)
        คืน Cycle เมื่อจบคัน (หรือครบเพดาน) ไม่งั้นคืน None"""
        step = row.get(self.step_key)
        if step is None:
            step = self.prev_step              # ไม่ทิ้งแถว: ถือว่า step เท่าเดิม

        completed = None
        if self.prev_step != 0 and step == 0:  # จุดตัด: คันเก่าจบ
            if self.synced and not self.capped:
                completed = self._emit(truncated=False)
            self.rows = []                     # ล้างก่อน -> 0 ตัวนี้เป็นแถวแรกของคันใหม่
            self.capped = False
            self.synced = True

        if self.synced and not self.capped:
            self.rows.append(dict(row))
            if len(self.rows) >= self.max_len:  # ครบเพดาน (สายหยุดค้างที่ step 2) -> ปล่อยเลย
                completed = self._emit(truncated=True)
                self.rows = []
                self.capped = True

        self.prev_step = step
        return completed

    def _emit(self, truncated: bool) -> Cycle | None:
        self.last_reject = None
        if not self.rows:
            return None
        ordered = sorted(self.rows, key=lambda r: r["recorded_at"])
        steps = [r.get(self.step_key) for r in ordered]
        reason = None
        if len(ordered) < self.min_len:
            reason = f"สั้นเกินไป ({len(ordered)} < {self.min_len} แถว)"
        elif steps[0] != 0:
            reason = f"ไม่ได้เริ่มที่ step 0 (เริ่มที่ {steps[0]})"
        elif 2 not in steps:
            reason = "ไม่ถึง step 2 (เครื่องหยุดกลางรอบ?)"
        if reason:
            self.last_reject = reason
            last = ordered[-1]
            log.warning("ข้ามรอบ %s bn=%s: %s", self.side, last.get(f"{self.side.lower()}_bn"), reason)
            return None

        last, p = ordered[-1], self.side.lower()
        return Cycle(
            side=self.side,
            bn=last.get(f"{p}_bn"),
            pgnum=last.get(f"{p}_pgnum"),
            asn=last.get(f"{p}_asn"),
            rows=[select_side_columns(r, self.side) for r in ordered],
            truncated=truncated,
        )
