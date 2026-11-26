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
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import time

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
DEDUPE_MARGIN = timedelta(minutes=5)        # เคยบันทึก (side, pgnum, bn) นี้ตั้งแต่ "เวลาที่รอบจบ - margin นี้" แล้ว = ไม่บันทึกซ้ำ
RETRY_MAX_SECONDS = 30                      # DB ล่ม: หน่วงเวลาลองใหม่ สูงสุดเท่านี้
DB_CALL_TIMEOUT = 60                        # เรียก DB (อ่าน/ล็อก) นานกว่านี้ = ถือว่าค้าง ข้ามไปลองรอบใหม่ (ไม่ให้ loop ค้างเงียบ)
STATUS_LOG_SECONDS = 60                     # log สถานะ poller ทุกกี่วินาที

# pgnum -> ฟังก์ชัน predict ของโมเดลนั้น
# โมเดลหนึ่งตัวใช้ได้ทั้ง LH และ RH : ฝั่ง RH จะถูกเปลี่ยนชื่อคอลัมน์ rh_* -> lh_* ก่อนส่งเข้าโมเดล (cycle.side_to_lh)
# เพิ่มรุ่นใหม่ = เติมบรรทัดเดียวที่นี่ (ฟังก์ชัน predict ต้องรับ df ที่มีคอลัมน์ lh_* เหมือน predict.run_prediction)
MODELS = {
    20: run_prediction,
    # 12: run_pg12,
}
SIDES = ("LH", "RH")

# สถานะของ poller (ดูได้ที่ GET /api/poller/status และใน log ทุก STATUS_LOG_SECONDS วินาที)
POLLER_STATUS = {
    "started_at": None,        # เวลาที่ poller เริ่ม
    "lock_held": False,        # ถือ advisory lock อยู่ไหม (False = รอ/เสีย lock ไม่ได้ประมวลผล)
    "cursor": None,            # id แถวสุดท้ายที่ประมวลผลแล้ว
    "last_row_at": None,       # recorded_at ของแถวสุดท้ายที่ประมวลผล (ใช้ดูว่าข้อมูลใหม่ล่าสุดเมื่อไหร่)
    "last_poll_ok": None,      # เวลา (unix) ที่ poll สำเร็จล่าสุด
    "last_error": None,        # ข้อความ error ล่าสุด (+ เวลา)
    "last_saved": None,        # รอบล่าสุดที่บันทึกสำเร็จ
    "counts": {"cycles": 0, "saved": 0, "skipped": 0, "errors": 0},
}


def _count(name: str) -> None:
    POLLER_STATUS["counts"][name] += 1


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
            _count("skipped")
            return
        if cycle.pgnum not in MODELS:
            log.info("skip: ไม่มีโมเดลสำหรับ %s", tag)
            _count("skipped")
            return
        # ผลของรอบนี้ถ้ามีอยู่แล้วต้องถูกบันทึก "หลังรอบจบ" -> เช็คตั้งแต่เวลาจบรอบ (ลบ margin กันนาฬิกาเครื่อง OPC กับ DB เหลื่อมกัน)
        # ใช้ได้ถูกต้องแม้ต้องประมวลผลย้อนหลังนานหลังหลุด และ bn ที่ถูกทำซ้ำคนละเวลาไม่ถูกนับว่าซ้ำ
        since = cycle.rows[-1]["recorded_at"] - DEDUPE_MARGIN
        if db.prediction_exists(cycle.side, cycle.pgnum, cycle.bn, since):
            log.info("skip: %s เคยบันทึกแล้ว", tag)
            _count("skipped")
            return

        result = predict_cycle(cycle)
        note = " (ถูกตัดที่เพดานความยาว)" if cycle.truncated else ""
        log.info("%s length=%s overall_pass=%s%s", tag, result["length"], result["overall_pass"], note)
        if POLLER_DRY_RUN:
            return
        try:
            db.save_result_from_dict(result)        # <-- เขียนลง DB อัตโนมัติทุกคันที่ predict เสร็จ
            _count("saved")
            POLLER_STATUS["last_saved"] = {"side": cycle.side, "pgnum": cycle.pgnum, "bn": cycle.bn,
                                           "at": datetime.now().astimezone().isoformat(timespec="seconds")}
        except Exception:                           # database ล่ม ไม่ให้ poller ตาย
            _count("errors")
            log.exception("save database ไม่สำเร็จ: %s", tag)
    except Exception:
        _count("errors")
        log.exception("predict ไม่สำเร็จ: %s", tag)


async def _db_call(fn, *args):
    """เรียกฟังก์ชัน DB (sync) ใน thread พร้อม timeout — ค้างเกิน DB_CALL_TIMEOUT = TimeoutError ให้ loop ข้ามไปลองใหม่
    (thread ที่ค้างจะจบเองเมื่อ keepalive/timeout ของ connection ทำงาน ดู db.CONNECT_KWARGS)"""
    return await asyncio.wait_for(asyncio.to_thread(fn, *args), DB_CALL_TIMEOUT)


def _log_status(cursor):
    st = POLLER_STATUS
    lag = None
    if st["last_row_at"] is not None:
        lag = max(0.0, (datetime.now(timezone.utc) - st["last_row_at"]).total_seconds())   # กันค่าติดลบจากนาฬิกาเครื่อง OPC เหลื่อมกับเครื่องนี้
    c = st["counts"]
    log.info("poller ปกติ: id=%s ข้อมูลล่าสุดห่างจากตอนนี้ %s | รอบที่เห็น=%d บันทึก=%d ข้าม=%d ผิดพลาด=%d",
             cursor, "-" if lag is None else f"{lag:.0f}s", c["cycles"], c["saved"], c["skipped"], c["errors"])


async def poll_loop():
    """ดึงแถวใหม่จากตาราง machine ทีละก้อน -> CycleBuffer -> predict + save  (cancel ได้ตอนปิดแอป)

    สถานะ (cursor + buffer ของรอบที่กำลังเดิน) อยู่ตลอดอายุ process: เสีย lock / DB หลุด แล้วกลับมา "ทำงานต่อจากจุดเดิม"
    รอบที่จบระหว่างที่หลุดจะถูกประมวลผลย้อนหลัง (ไม่ข้าม) — warm-up (replay เงียบ) ทำครั้งเดียวตอนเริ่ม process เท่านั้น
    ถ้าระหว่างที่เสีย lock มี poller ตัวอื่นประมวลผลรอบเดียวกันไปแล้ว prediction_exists จะกันไม่ให้บันทึกซ้ำ"""
    st = POLLER_STATUS
    buffers = {s: CycleBuffer(s) for s in SIDES}            # buffer ต่อข้าง: LH และ RH ใช้โมเดลเดียวกันตาม pgnum
    cursor = None                                           # id แถวสุดท้ายที่ประมวลผลแล้ว (None = ยังไม่เคย warm-up)
    lock = db.PollerLock()
    delay = POLL_SECONDS
    last_status_log = 0.0
    st["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    log.info("poller เริ่มทำงาน sides=%s dry_run=%s every=%ss", list(SIDES), POLLER_DRY_RUN, POLL_SECONDS)
    try:
        while True:
            try:
                # ถือ advisory lock ตลอด: มี poller ได้ตัวเดียว (ตัวอื่นรอรับช่วงต่อถ้าตัวนี้ตาย)
                if not await _db_call(lock.held):
                    if st["lock_held"]:
                        log.warning("เสีย advisory lock (connection หลุด) — หยุดประมวลผลชั่วคราว รอต่อ lock ใหม่")
                    st["lock_held"] = False
                    if not await _db_call(lock.acquire):
                        log.info("มี poller อื่นทำงานอยู่แล้ว (advisory lock) — รอ")
                        await asyncio.sleep(RETRY_MAX_SECONDS)
                        continue
                    log.info("ได้ advisory lock แล้ว%s",
                             "" if cursor is None else f" — ทำงานต่อจาก id > {cursor} (รอบที่จบระหว่างหลุดจะถูกประมวลผลย้อนหลัง)")
                st["lock_held"] = True

                if cursor is None:
                    # เริ่ม process: replay แถวท้ายตารางแบบเงียบ (ไม่ predict) ให้รอบที่กำลังเดินอยู่ครบ — ไม่ย้อน predict รอบเก่า
                    tail = await _db_call(db.get_machine_tail, WARMUP_ROWS)
                    for row in tail:
                        for buf in buffers.values():
                            buf.update(row)
                    cursor = tail[-1]["id"] if tail else 0
                    if tail:
                        st["last_row_at"] = tail[-1]["recorded_at"]
                    log.info("warm-up เสร็จ: replay %d แถว, เริ่มดึงต่อจาก id > %s", len(tail), cursor)

                rows = await _db_call(db.get_machine_rows_after, cursor, BATCH_ROWS)
                for row in rows:
                    if row["id"] != cursor + 1:
                        log.warning("id ข้ามช่วง: %s -> %s", cursor, row["id"])
                    cursor = row["id"]
                    st["last_row_at"] = row["recorded_at"]
                    for buf in buffers.values():
                        cycle = buf.update(row)
                        if cycle is not None:
                            _count("cycles")
                            await asyncio.to_thread(process_cycle, cycle)   # ทีละรอบ ไม่บล็อก event loop (ไม่ใส่ timeout: ห้ามตัดกลางคันตอนบันทึก)

                st["cursor"] = cursor
                st["last_poll_ok"] = time.time()
                delay = 0 if len(rows) == BATCH_ROWS else POLL_SECONDS  # ยังมีค้างอยู่ -> ดึงต่อทันที
                if time.monotonic() - last_status_log >= STATUS_LOG_SECONDS:
                    last_status_log = time.monotonic()
                    _log_status(cursor)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _count("errors")
                st["last_error"] = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {type(e).__name__}: {e}"[:300]
                delay = min(max(delay, 1) * 2, RETRY_MAX_SECONDS)       # DB ล่ม ฯลฯ -> หน่วงเวลาแล้วลองใหม่ (สถานะเดิมยังอยู่)
                log.exception("poll ไม่สำเร็จ ลองใหม่ใน %ss", delay)
            await asyncio.sleep(delay)
    finally:
        st["lock_held"] = False
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


@api.get("/poller/status")
def poller_status():
    """สถานะ poller (ดึงข้อมูล -> predict -> บันทึก) ไว้เช็คว่ายังทำงานอยู่ไหม
    healthy = poll สำเร็จเมื่อไม่นานนี้ + ถือ lock อยู่ ; data_lag_seconds = ข้อมูลใน machine ใหม่ล่าสุดห่างจากตอนนี้กี่วินาที
    (healthy แต่ data_lag สูง = poller ปกติ แต่ตัวเก็บข้อมูลไม่ส่งแถวใหม่เข้ามา)"""
    st = POLLER_STATUS
    now = time.time()
    last_ok = st["last_poll_ok"]
    poll_age = None if last_ok is None else round(now - last_ok, 1)
    last_row_at = st["last_row_at"]
    lag = None if last_row_at is None else round(max(0.0, (datetime.now(timezone.utc) - last_row_at).total_seconds()), 1)
    alive = poll_age is not None and poll_age <= max(30.0, POLL_SECONDS * 5)
    return {
        "enabled": POLLER_ENABLED,
        "dry_run": POLLER_DRY_RUN,
        "healthy": bool(POLLER_ENABLED and alive and st["lock_held"]),
        "lock_held": st["lock_held"],
        "started_at": st["started_at"],
        "last_poll_seconds_ago": poll_age,
        "cursor": st["cursor"],
        "last_row_at": None if last_row_at is None else last_row_at.isoformat(),
        "data_lag_seconds": lag,
        "counts": dict(st["counts"]),
        "last_saved": st["last_saved"],
        "last_error": st["last_error"],
    }


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