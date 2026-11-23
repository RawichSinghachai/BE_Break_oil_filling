"""
main.py — ดึงข้อมูลจากตาราง machine -> ประกอบเป็นรถทีละคัน -> predict -> บันทึก database อัตโนมัติ -> ให้ดึงผลผ่าน API

flow: poller (background task) ดึงแถวใหม่จากตาราง machine ทุก POLL_SECONDS วินาที -> cycle.CycleBuffer ตัดเป็นรอบของรถแต่ละคัน
      (ตัดตอน step ไม่ใช่ 0 -> 0) -> predict -> save_result ลง DB เอง
API มีไว้ "ดึงผล" อย่างเดียว (ไม่มี endpoint ยิง predict เอง)

ติดตั้ง:  pip install fastapi uvicorn
รัน:      uvicorn main:app --host 0.0.0.0 --port 8001      (อย่าใช้ --reload; ถ้ารันหลาย worker จะมี poller ทำงานแค่ตัวเดียว ด้วย advisory lock)
เอกสาร API (ลองยิงได้ในเว็บ):  http://<ip>:8001/docs

ตัวแปร environment (ไม่บังคับ):
  POLLER_ENABLED=0   ปิด poller (รันแค่ API) — ใช้ตอนที่ยังมี API ตัวเก่าทำหน้าที่ predict อยู่
  POLLER_DRY_RUN=1   poller predict แล้ว log ผลอย่างเดียว ไม่บันทึกลง DB
  POLL_SECONDS       ความถี่ดึงข้อมูล (ค่าเริ่มต้น 2)
"""
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, APIRouter, Query
from fastapi.responses import HTMLResponse


import pandas as pd

import db                                   # save_result_from_dict / get_predictions / get_prediction_by_bn / get_latest_filling
from cycle import CycleBuffer, Cycle, side_to_lh
from predict import run_prediction        # โมเดล PGNum 20 (ใช้ได้ทั้ง LH และ RH)
# from predict_pg12 import run_prediction as run_pg12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("poller")

POLLER_ENABLED = os.getenv("POLLER_ENABLED", "1") != "0"
POLLER_DRY_RUN = os.getenv("POLLER_DRY_RUN", "0") == "1"
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))
BATCH_ROWS = 5000                           # จำนวนแถวสูงสุดต่อการดึง 1 ครั้ง (ดึงต่อทันทีถ้ายังมีค้าง)
WARMUP_ROWS = 300                           # ตอนเริ่มระบบ replay แถวท้ายตารางเท่านี้ เพื่อให้รอบที่กำลังเดินอยู่ครบ
DEDUPE_WINDOW = timedelta(hours=1)          # เคยบันทึก (side, pgnum, bn) นี้ภายในช่วงนี้แล้ว = ไม่บันทึกซ้ำ
RETRY_MAX_SECONDS = 30                      # DB ล่ม: หน่วงเวลาลองใหม่ สูงสุดเท่านี้

# pgnum -> ฟังก์ชัน predict ของโมเดลนั้น
# โมเดลหนึ่งตัวใช้ได้ทั้ง LH และ RH : ฝั่ง RH จะถูกเปลี่ยนชื่อคอลัมน์ rh_* -> lh_* ก่อนส่งเข้าโมเดล (cycle.side_to_lh)
# เพิ่มรุ่นใหม่ = เติมบรรทัดเดียวที่นี่ (ฟังก์ชัน predict ต้องรับ df ที่มีคอลัมน์ lh_* เหมือน predict.run_prediction)
MODELS = {
    20: run_prediction,
    # 12: run_pg12,
}
SIDES = ("LH", "RH")


def predict_cycle(cycle: Cycle):
    """predict + เตรียมผลให้พร้อมบันทึก (sync, หนัก: เรียกผ่าน asyncio.to_thread)
    คืน result หรือ None ถ้าไม่มีโมเดลของ pgnum นี้"""
    predict_fn = MODELS.get(cycle.pgnum)
    if predict_fn is None:
        return None

    df = pd.DataFrame([side_to_lh(r, cycle.side) for r in cycle.rows])
    df.columns = df.columns.str.lower()
    df = df.sort_values("recorded_at")
    df["id"] = range(len(df))

    result = predict_fn(df)
    return {"side": cycle.side, "pgnum": cycle.pgnum, "asn": cycle.asn,
            "predicted_at": datetime.now().isoformat(timespec="seconds"),
            **result}


def process_cycle(cycle: Cycle):
    """รอบที่จบแล้ว 1 รอบ -> predict -> บันทึก (sync: เรียกผ่าน asyncio.to_thread)  ไม่ raise ออกไป: รอบเดียวพังไม่ทำให้ poller หยุด"""
    tag = f"{cycle.side} PG{cycle.pgnum} BN {cycle.bn}"
    try:
        if cycle.bn is None or cycle.pgnum is None:
            log.warning("ข้ามรอบ %s: ไม่มี bn/pgnum", tag)
            return
        if cycle.pgnum not in MODELS:
            log.info("skip: ไม่มีโมเดลสำหรับ %s", tag)
            return
        if db.prediction_exists(cycle.side, cycle.pgnum, cycle.bn, datetime.now().astimezone() - DEDUPE_WINDOW):
            log.info("skip: %s เคยบันทึกแล้ว", tag)
            return

        result = predict_cycle(cycle)
        note = " (ถูกตัดที่เพดานความยาว)" if cycle.truncated else ""
        log.info("%s length=%s overall_pass=%s%s", tag, result["length"], result["overall_pass"], note)
        if POLLER_DRY_RUN:
            return
        try:
            db.save_result_from_dict(result)        # <-- เขียนลง DB อัตโนมัติทุกคันที่ predict เสร็จ
        except Exception:                           # database ล่ม ไม่ให้ poller ตาย
            log.exception("save database ไม่สำเร็จ: %s", tag)
    except Exception:
        log.exception("predict ไม่สำเร็จ: %s", tag)


async def poll_loop():
    """ดึงแถวใหม่จากตาราง machine ทีละก้อน -> CycleBuffer -> predict + save  (cancel ได้ตอนปิดแอป)"""
    sides = list(SIDES)                                     # buffer ต่อข้าง: LH และ RH ใช้โมเดลเดียวกันตาม pgnum
    buffers: dict[str, CycleBuffer] = {}
    cursor = None                                           # id แถวสุดท้ายที่ประมวลผลแล้ว (None = ยังไม่ได้ warm-up)
    lock = db.PollerLock()
    delay = POLL_SECONDS

    log.info("poller เริ่มทำงาน sides=%s dry_run=%s every=%ss", sides, POLLER_DRY_RUN, POLL_SECONDS)
    try:
        while True:
            try:
                # ถือ advisory lock ตลอด: มี poller ได้ตัวเดียว (ตัวอื่นรอรับช่วงต่อถ้าตัวนี้ตาย)
                if not await asyncio.to_thread(lock.held):
                    if not await asyncio.to_thread(lock.acquire):
                        log.info("มี poller อื่นทำงานอยู่แล้ว (advisory lock) — รอ")
                        await asyncio.sleep(RETRY_MAX_SECONDS)
                        continue
                    cursor = None                           # ได้ล็อกใหม่ -> warm-up ใหม่ ไม่ใช้สถานะเก่าที่อาจค้าง

                if cursor is None:
                    # เริ่มระบบ: replay แถวท้ายตารางแบบเงียบ (ไม่ predict) ให้รอบที่กำลังเดินอยู่ครบ — ไม่ย้อน predict รอบเก่า
                    tail = await asyncio.to_thread(db.get_machine_tail, WARMUP_ROWS)
                    buffers = {s: CycleBuffer(s) for s in sides}
                    for row in tail:
                        for buf in buffers.values():
                            buf.update(row)
                    cursor = tail[-1]["id"] if tail else 0
                    log.info("warm-up เสร็จ: replay %d แถว, เริ่มดึงต่อจาก id > %s", len(tail), cursor)

                rows = await asyncio.to_thread(db.get_machine_rows_after, cursor, BATCH_ROWS)
                for row in rows:
                    if row["id"] != cursor + 1:
                        log.warning("id ข้ามช่วง: %s -> %s", cursor, row["id"])
                    cursor = row["id"]
                    for buf in buffers.values():
                        cycle = buf.update(row)
                        if cycle is not None:
                            await asyncio.to_thread(process_cycle, cycle)   # ทีละรอบ ไม่บล็อก event loop

                delay = 0 if len(rows) == BATCH_ROWS else POLL_SECONDS  # ยังมีค้างอยู่ -> ดึงต่อทันที
            except asyncio.CancelledError:
                raise
            except Exception:
                delay = min(max(delay, 1) * 2, RETRY_MAX_SECONDS)       # DB ล่ม ฯลฯ -> หน่วงเวลาแล้วลองใหม่
                log.exception("poll ไม่สำเร็จ ลองใหม่ใน %ss", delay)
            await asyncio.sleep(delay)
    finally:
        lock.release()


# ---------- API (ดึงผลอย่างเดียว) ----------

@asynccontextmanager
async def lifespan(app):
    db.pool.open()                          # 1) เปิด connection pool (sync) ก่อน
    poller = asyncio.create_task(poll_loop()) if POLLER_ENABLED else None   # 2) ค่อยเริ่ม poller
    if poller is None:
        log.info("POLLER_ENABLED=0 -> ไม่เปิด poller (รันแค่ API)")
    yield
    if poller is not None:                  # ปิดตามลำดับย้อนกลับ: poller ก่อน แล้วค่อย DB
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass
    db.pool.close()



app = FastAPI(title="Brake Oil Filling Anomaly API", lifespan=lifespan)
api = APIRouter(prefix="/api")

@api.websocket("/ws/realtime")
async def realtime(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # db.get_latest_filling เป็น sync -> รันใน thread แยก ไม่บล็อก event loop
            row = await asyncio.to_thread(db.get_latest_filling)

            if row:
                await ws.send_json({
                    "recorded_at": row["recorded_at"].isoformat() if row["recorded_at"] else None,
                    "lh": {
                        "bn":               row["lh_bn"],
                        "pgnum":            row["lh_pgnum"],
                        "stepnum":          row["lh_stepnum"],
                        "filling_pressure": row["lh_fillingpressure"],
                        "filling_volume":   row["lh_fillingvolume"],
                        "vacuumgun":        row["lh_vacuumgun"],
                    },
                    "rh": {
                        "bn":               row["rh_bn"],
                        "pgnum":            row["rh_pgnum"],
                        "stepnum":          row["rh_stepnum"],
                        "filling_pressure": row["rh_fillingpressure"],
                        "filling_volume":   row["rh_fillingvolume"],
                        "vacuumgun":        row["rh_vacuumgun"],
                    },
                    "common": {
                        "continue":         row["continue"],
                        "master_on":        row["master_on"],
                        "machine_fault":    row["machine_fault"],
                    },
                })

            await asyncio.sleep(1)       # ส่งทุก 1 วิ ตามจังหวะที่ OPC เก็บ
    except WebSocketDisconnect:
        pass


@api.get("")
def root():
    """เช็คว่า API ทำงานอยู่ — GET /api -> {"message": "Hello World"}"""
    return {"message": "Hello World"}


@api.get("/predictions")
def list_predictions(
    side: str | None = Query(None, description="LH หรือ RH"),
    pgnum: int | None = Query(None, description="เลข PG เช่น 20"),
    bn: int | None = Query(None, description="เลข BN"),
    asn: int | None = Query(None, description="เลข ASN"),
    overall_pass: bool | None = Query(None, description="true = ผ่าน, false = NG"),
    dt_from: datetime | None = Query(None, description="predicted_at >= (ISO 8601 เช่น 2026-11-04T00:00:00+07:00)"),
    dt_to: datetime | None = Query(None, description="predicted_at <= (ISO 8601)"),
    limit: int | None = Query(None, ge=1, description="ไม่ส่ง = ดึงทั้งหมด (frontend แบ่งหน้าเอง)"),
    offset: int = Query(0, ge=0),
):
    """รายการผลตรวจ ใหม่สุดอยู่บน (ไม่รวม input / recon) — ใช้กับตารางประวัติ, KPI, NG แยก pgnum
    ทุก parameter เป็น optional filter เช่น /predictions?side=LH&pgnum=20&overall_pass=false&limit=50"""
    if side is not None:
        side = side.upper()
        if side not in ("LH", "RH"):
            raise HTTPException(422, "side ต้องเป็น 'LH' หรือ 'RH'")
    # db.get_predictions เป็น sync -> ประกาศ endpoint เป็น def (ไม่ใช่ async def)
    # FastAPI จะรันใน threadpool ให้เอง ไม่บล็อก event loop
    return db.get_predictions(
        side=side,
        pgnum=None if pgnum is None else str(pgnum),
        bn=None if bn is None else str(bn),
        asn=None if asn is None else str(asn),
        overall_pass=overall_pass,
        dt_from=dt_from,
        dt_to=dt_to,
        limit=limit,
        offset=offset,
    )


@api.get("/predictions/{bn}")
def prediction_by_bn(bn: int):
    """ผลตรวจแบบเต็ม (รวม input / recon สำหรับพล็อตกราฟ) ของ BN ที่ระบุ
    bn ซ้ำได้ -> คืนแถวล่าสุดแถวเดียว, ไม่พบ -> 404"""
    row = db.get_prediction_by_bn(str(bn))
    if row is None:
        raise HTTPException(404, f"ไม่พบผลตรวจของ BN {bn}")
    return row


@api.get("/machines")
def list_machines(
    side: str | None = Query(None, description="LH หรือ RH"),
    pgnum: int | None = Query(None, description="เลข PG เช่น 20"),
    bn: int | None = Query(None, description="เลข BN"),
    asn: int | None = Query(None, description="เลข ASN"),
    dt_from: datetime | None = Query(None, description="recorded_at >= (ISO 8601 เช่น 2026-11-04T00:00:00+07:00)"),
    dt_to: datetime | None = Query(None, description="recorded_at <= (ISO 8601)"),
    limit: int = Query(50, ge=1, le=1000, description="จำนวนแถวต่อหน้า (page size)"),
    offset: int = Query(0, ge=0, description="ข้ามกี่แถว (หน้า N = (N-1) * limit)"),
):
    """ค่า real-time ย้อนหลังจากตาราง machine (1 แถวต่อข้าง) เรียงตาม recorded_at ใหม่สุดอยู่บน
    คืน {items, total, limit, offset, page, total_pages}
    ทุก filter เป็น optional เช่น /machines?side=LH&pgnum=20&limit=50&offset=100"""
    if side is not None:
        side = side.upper()
        if side not in ("LH", "RH"):
            raise HTTPException(422, "side ต้องเป็น 'LH' หรือ 'RH'")
    filters = dict(
        side=side,
        pgnum=None if pgnum is None else str(pgnum),
        bn=None if bn is None else str(bn),
        asn=None if asn is None else str(asn),
        dt_from=dt_from,
        dt_to=dt_to,
    )
    # db.* เป็น sync -> ประกาศ endpoint เป็น def (ไม่ใช่ async def)
    total = db.count_machines(**filters)
    items = db.get_machines(**filters, limit=limit, offset=offset)
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "page": offset // limit + 1,
        "total_pages": -(-total // limit),    # ceil(total / limit); total = 0 -> 0
    }


@api.get("/machines/{bn}")
def machines_by_bn(bn: int):
    """ข้อมูลจากตาราง machine ทุกคอลัมน์ ของ BN ที่ระบุ (ค้นทั้ง LH และ RH) ใหม่สุดอยู่บน
    bn ซ้ำได้ / เก็บทุก ~1 วิ -> คืนทุกแถวที่ตรง (ไม่จำกัดจำนวน), ไม่พบ -> 404"""
    # db.get_machines_by_bn เป็น sync -> ประกาศ endpoint เป็น def (ไม่ใช่ async def)
    rows = db.get_machines_by_bn(str(bn))
    if not rows:
        raise HTTPException(404, f"ไม่พบข้อมูลเครื่องของ BN {bn}")
    return rows


app.include_router(api)   # ต้องอยู่หลังประกาศ route ทั้งหมดของ api

# @app.get("/models")
# def list_models():
#     """โมเดลที่ใช้ได้"""
#     return [{"side": s, "pgnum": p} for s, p in MODELS]


# @app.get("/results")
# def list_results(side: str = None, pgnum: int = None, overall_pass: bool = None, limit: int = 50):
#     """ผลล่าสุดแบบย่อ ใหม่สุดอยู่บน กรองได้ เช่น /results?side=LH&pgnum=20&overall_pass=false"""
#     return db.get_results(side=side.upper() if side else None, pgnum=pgnum,
#                           overall_pass=overall_pass, limit=limit)


# @app.get("/results/latest")
# def latest_result(side: str = "LH", pgnum: int = 20):
#     """ผลเต็มของคันล่าสุด (มี input/recon สำหรับพล็อตกราฟ)"""
#     r = db.get_latest(side.upper(), pgnum)
#     if r is None:
#         raise HTTPException(404, f"ยังไม่มีผลของ {side.upper()} PG{pgnum}")
#     return r


# @app.get("/results/{bn}")
# def result_by_bn(bn: int, side: str = "LH", pgnum: int = 20):
#     """ผลเต็มของรถคันที่ระบุ BN"""
#     r = db.get_by_bn(side.upper(), pgnum, bn)
#     if r is None:
#         raise HTTPException(404, f"ไม่พบ BN {bn} ({side.upper()} PG{pgnum})")
#     return r