"""
main.py — รับข้อมูลจาก MQTT -> filter side/pgnum -> predict -> บันทึก database อัตโนมัติ -> ให้ดึงผลผ่าน API

flow: batch รถมาจาก MQTT (topic TOPIC_IN) -> predict -> save_result ลง DB เอง -> publish ผลออก TOPIC_OUT
API มีไว้ "ดึงผล" อย่างเดียว (ไม่มี endpoint ยิง predict เอง — ใช้ MQTT เป็น trigger ทางเดียว)

ติดตั้ง:  pip install fastapi uvicorn "paho-mqtt>=2"
รัน:      uvicorn main:app --host 0.0.0.0 --port 8001      (อย่าใช้ --reload หรือ --workers > 1)
เอกสาร API (ลองยิงได้ในเว็บ):  http://<ip>:8001/docs
"""
import json
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, APIRouter, Query
from fastapi.responses import HTMLResponse


import pandas as pd
import paho.mqtt.client as mqtt

import db                                   # save_result_from_dict / get_predictions / get_prediction_by_bn / get_latest_filling
from predict import run_prediction        # โมเดล LH / PGNum 20
# from predict_lh12 import run_prediction as run_lh12
# from predict_rh20 import run_prediction as run_rh20

BROKER = "100.124.242.18"
PORT = 1883
TOPIC_IN = "tmt/banpho/assembly/final3/filling_equipment/BFC/raw_data/LH/20"
TOPIC_OUT = "filling/result"

# (side, pgnum) -> ฟังก์ชัน predict ของโมเดลนั้น
MODELS = {
    ("LH", 20): run_prediction,
    # ("LH", 12): run_lh12,
    # ("RH", 20): run_rh20,
}


def predict_message(data):
    """filter + เตรียมข้อมูล + predict + บันทึก database อัตโนมัติ
    คืน result หรือ None ถ้าไม่มีโมเดล"""
    side = str(data["side"]).upper()
    pgnum = int(data["pgnum"])
    predict_fn = MODELS.get((side, pgnum))
    if predict_fn is None:
        return None

    df = pd.DataFrame(data["payload"])
    df.columns = df.columns.str.lower()
    df = df.sort_values("recorded_at")
    df["id"] = range(len(df))

    result = predict_fn(df)

    result = {"side": side, "pgnum": pgnum,
              "predicted_at": datetime.now().isoformat(timespec="seconds"),
              **result}

    try:
        db.save_result_from_dict(result)     # <-- เขียนลง DB อัตโนมัติทุกคันที่ predict เสร็จ
    except Exception as e:                  # database ล่ม ไม่ให้ผล predict หาย
        print("save database ไม่สำเร็จ:", e)

    return result


# ---------- MQTT (trigger หลัก: batch รถเข้ามา -> predict -> save) ----------
def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT connected, subscribe", TOPIC_IN)
    client.subscribe(TOPIC_IN)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        result = predict_message(data)
    except Exception as e:                  # ข้อความเสีย 1 ชิ้นไม่ทำให้ระบบหยุด
        print("error:", e)
        return
    if result is None:
        print(f"skip: ไม่มีโมเดลสำหรับ {data.get('side')} PG{data.get('pgnum')}")
        return
    try:                                    # publish พังไม่ทำให้ callback / thread MQTT ล่ม
        print(f"{result['side']} PG{result['pgnum']} BN {result['bn']} "
              f"overall_pass={result['overall_pass']}")
        # _json_safe: numpy -> python, NaN/Inf -> None (ให้เป็น JSON ที่ valid)
        payload = json.dumps(db._json_safe(result), allow_nan=False)
        client.publish(f"{TOPIC_OUT}/{result['side']}/{result['pgnum']}", payload, qos=1)
    except Exception as e:
        print("publish ไม่สำเร็จ:", e)


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
# ยังไม่ connect ตรงนี้ — ไปเริ่มใน lifespan หลัง db.pool.open() เพื่อไม่ให้ข้อความเข้ามาก่อน DB พร้อม


# ---------- API (ดึงผลอย่างเดียว) ----------

@asynccontextmanager
async def lifespan(app):
    db.pool.open()                          # 1) เปิด connection pool (sync) ก่อน
    mqtt_client.connect_async(BROKER, PORT) # 2) ค่อยเริ่ม MQTT (ไม่ค้างถ้า broker ยังไม่พร้อม จะต่อใหม่ให้เอง)
    mqtt_client.loop_start()                #    MQTT ทำงานใน thread แยก
    yield
    mqtt_client.loop_stop()                 # ปิดตามลำดับย้อนกลับ: MQTT ก่อน แล้วค่อย DB
    mqtt_client.disconnect()
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
                        "filling_pressure": row["lh_fillingpressure"],
                        "filling_volume":   row["lh_fillingvolume"],
                        "vacuumgun":        row["lh_vacuumgun"],
                    },
                    "rh": {
                        "bn":               row["rh_bn"],
                        "pgnum":            row["rh_pgnum"],
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