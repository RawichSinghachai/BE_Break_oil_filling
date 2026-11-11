import math
import os

from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# ---- ตั้งค่าการเชื่อมต่อ (Postgres บนเครื่อง Windows ผ่าน Tailscale) ----
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "postgres"),
}

# make_conninfo จัดการอักขระพิเศษใน password (เช่น @) ให้เอง ไม่ต้อง URL-encode
DATABASE_URL = os.getenv("DATABASE_URL") or make_conninfo(**DB_CONFIG)

# ---- ชื่อตาราง (แก้ให้ตรงกับ DB จริง) ----
PRED_TABLE = "prediction_break_oil_filling_machine"
MACHINE_TABLE = "machine"

# ---- connection pool (เปิดครั้งเดียว ใช้ซ้ำทั้งแอป) ----
pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"row_factory": dict_row},  # ผลลัพธ์เป็น dict; JSONB แปลงเป็น dict/list ให้อัตโนมัติ
)

def get_latest_filling() -> dict | None:
    """ค่า real-time แถวล่าสุดจากตาราง machine :
    filling_pressure / filling_volume / vacuumgun ทั้ง LH, RH (+ recorded_at)
    คืน dict (key ตัวพิมพ์เล็ก) หรือ None ถ้ายังไม่มีข้อมูล"""
    sql = """
        SELECT Continue, Master_on, Machine_fault, LH_BN, LH_PgNum, LH_FillingPressure, LH_FillingVolume, LH_VacuumGun,
            RH_BN, RH_PgNum, RH_FillingPressure, RH_FillingVolume, RH_VacuumGun,
            Recorded_at
        FROM machine
        ORDER BY id DESC
        LIMIT 1
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()

        
def get_predictions(
    side: str | None = None,
    pgnum: str | None = None,
    bn: str | None = None,
    asn: str | None = None,
    overall_pass=None,
    dt_from=None,
    dt_to=None,
    limit=None,
    offset=0,
) -> list[dict]:
    """
    ดึงรายการผลตรวจ (ใช้กับ: ตารางประวัติ, KPI วันนี้, NG แยก pgnum)
    - ไม่ดึง input / recon (ก้อนใหญ่) — เอาแค่ที่ตาราง/กราฟสรุปต้องใช้
    - features (JSONB) ดึงมาทั้งก้อน ให้ frontend อ่าน percent/pass เอง
    - limit=None → LIMIT NULL = ดึงทั้งหมด (frontend แบ่งหน้าเอง)
    ทุก argument เป็น optional filter: ส่ง None = ไม่ filter ตัวนั้น
    bn / asn / pgnum เป็น str
    """
    sql = """
        SELECT id, side, pgnum, bn, asn, predicted_at, length,
               overall_pass, features
        FROM prediction_break_oil_filling_machine
        WHERE (%(side)s::text         IS NULL OR side = %(side)s::text)
          AND (%(pgnum)s::text        IS NULL OR pgnum::text = %(pgnum)s::text)
          AND (%(bn)s::text           IS NULL OR bn::text = %(bn)s::text)
          AND (%(asn)s::text          IS NULL OR asn::text = %(asn)s::text)
          AND (%(overall_pass)s::boolean   IS NULL OR overall_pass = %(overall_pass)s::boolean)
          AND (%(dt_from)s::timestamptz    IS NULL OR predicted_at >= %(dt_from)s::timestamptz)
          AND (%(dt_to)s::timestamptz      IS NULL OR predicted_at <= %(dt_to)s::timestamptz)
        ORDER BY predicted_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    params = {
        "side": side,
        "pgnum": pgnum,
        "bn": bn,
        "asn": asn,
        "overall_pass": overall_pass,
        "dt_from": dt_from,
        "dt_to": dt_to,
        "limit": limit,
        "offset": offset,
    }
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
 
 
def get_prediction_by_bn(bn: str) -> dict | None:
    """
    ดึงผลตรวจแบบเต็ม รวม input / recon ของ bn ที่ระบุ (ใช้กับ modal รายละเอียดรถ)
    bn ซ้ำได้ (append-only) → คืน "แถวล่าสุด" แถวเดียว
    คืน None ถ้าไม่พบ
    """
    sql = """
        SELECT id, side, pgnum, bn, asn, predicted_at, length,
               overall_pass, features, input, recon
        FROM prediction_break_oil_filling_machine
        WHERE bn::text = %(bn)s::text
        ORDER BY predicted_at DESC
        LIMIT 1
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"bn": bn})
            return cur.fetchone()


# ตาราง machine เก็บ LH / RH อยู่แถวเดียวกัน → แตกเป็น 1 แถวต่อข้าง (side) ใช้ร่วมกันทั้ง get / count
_MACHINES_BY_SIDE = """
        SELECT id, 'LH'::text AS side,
               LH_PgNum AS pgnum, LH_BN AS bn, LH_ASN AS asn,
               LH_FillingPressure AS filling_pressure,
               LH_FillingVolume   AS filling_volume,
               LH_VacuumGun       AS vacuumgun,
               Recorded_at        AS recorded_at
        FROM machine
        UNION ALL
        SELECT id, 'RH'::text AS side,
               RH_PgNum, RH_BN, RH_ASN,
               RH_FillingPressure, RH_FillingVolume, RH_VacuumGun,
               Recorded_at
        FROM machine
"""


def _machines_where(side, pgnum, bn, asn, dt_from, dt_to):
    """สร้าง WHERE + params เฉพาะ filter ที่ส่งมา (ใช้ร่วมกันระหว่าง get_machines / count_machines)
    เทียบ pgnum / bn / asn ตรง ๆ (ไม่ ::text) ให้ Postgres แปลงเป็นชนิดของคอลัมน์เอง"""
    conditions, params = [], {}
    if side is not None:
        conditions.append("side = %(side)s")
        params["side"] = side
    if pgnum is not None:
        conditions.append("pgnum = %(pgnum)s")
        params["pgnum"] = pgnum
    if bn is not None:
        conditions.append("bn = %(bn)s")
        params["bn"] = bn
    if asn is not None:
        conditions.append("asn = %(asn)s")
        params["asn"] = asn
    if dt_from is not None:
        conditions.append("recorded_at >= %(dt_from)s::timestamptz")
        params["dt_from"] = dt_from
    if dt_to is not None:
        conditions.append("recorded_at <= %(dt_to)s::timestamptz")
        params["dt_to"] = dt_to
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


def get_machines(
    side: str | None = None,
    pgnum: str | None = None,
    bn: str | None = None,
    asn: str | None = None,
    dt_from=None,
    dt_to=None,
    limit=1000,
    offset=0,
) -> list[dict]:
    """
    ดึงรายการค่า real-time จากตาราง machine (ใช้กับ: ตารางประวัติค่าเครื่อง, กราฟย้อนหลัง)
    - 1 แถวต่อข้าง (side) เหมือน get_predictions
    - คอลัมน์ที่คืน: id, side, pgnum, bn, asn, filling_pressure, filling_volume, vacuumgun, recorded_at
    - เรียงตาม recorded_at DESC (ใหม่สุดอยู่บน) + id DESC, side เป็น tie-breaker → แบ่งหน้าแล้วลำดับนิ่ง
    - limit=None → LIMIT NULL = ดึงทั้งหมด (ระวัง: ตารางนี้เก็บทุก ~1 วิ แถวเยอะมาก)
    ทุก argument เป็น optional filter: ส่ง None = ไม่ filter ตัวนั้น
    bn / asn / pgnum เป็น str
    """
    where, params = _machines_where(side, pgnum, bn, asn, dt_from, dt_to)
    params.update({"limit": limit, "offset": offset})
    query = f"""
        SELECT id, side, pgnum, bn, asn,
               filling_pressure, filling_volume, vacuumgun, recorded_at
        FROM ({_MACHINES_BY_SIDE}) m
        {where}
        ORDER BY recorded_at DESC, id DESC, side
        LIMIT %(limit)s OFFSET %(offset)s
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def count_machines(
    side: str | None = None,
    pgnum: str | None = None,
    bn: str | None = None,
    asn: str | None = None,
    dt_from=None,
    dt_to=None,
) -> int:
    """จำนวนแถวทั้งหมดที่ตรง filter (filter ชุดเดียวกับ get_machines) — ใช้คำนวณ total_pages"""
    where, params = _machines_where(side, pgnum, bn, asn, dt_from, dt_to)
    query = f"""
        SELECT COUNT(*) AS total
        FROM ({_MACHINES_BY_SIDE}) m
        {where}
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()["total"]


def get_machines_by_bn(bn: str) -> list[dict]:
    """
    ดึงข้อมูลจากตาราง machine ของ bn ที่ระบุ — "ทุกคอลัมน์" ที่มีในตาราง (SELECT *)
    - ค้นทั้ง LH_BN และ RH_BN : แถวที่ lh_bn = bn หรือ rh_bn = bn
    - 1 แถวในตาราง = 1 แถวผลลัพธ์ (ไม่แตกเป็นรายข้าง) มีทั้ง lh_* / rh_* และค่าร่วม
      (continue / master_on / machine_fault) ครบ, key เป็นตัวพิมพ์เล็ก
    - ใหม่สุดอยู่บน (ORDER BY recorded_at DESC)
    - bn ซ้ำได้ (เก็บทุก ~1 วิ → รถคันเดียวมีหลายแถว) → คืนทุกแถวที่ตรง, limit=None = ไม่จำกัด
    - เทียบ bn ตรง ๆ (ไม่ ::text) ให้ Postgres แปลงเป็นชนิดของคอลัมน์เอง
    bn เป็น str
    """
    sql = """
        SELECT *
        FROM machine
        WHERE lh_bn = %(bn)s OR rh_bn = %(bn)s
        ORDER BY recorded_at DESC, id DESC
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"bn": bn})
            return cur.fetchall()

# ======================================================================
# WRITE: บันทึกผล predict ลงตาราง prediction_break_oil_filling_machine
# ======================================================================
def _json_safe(obj):
    """แปลงข้อมูลให้ JSONB รับได้:
    - numpy array / numpy scalar → list / ค่า python ธรรมดา
    - NaN / Inf → None (Postgres JSONB ไม่รับ NaN, json.dumps ปกติจะสร้าง NaN ออกมา)
    - key ของ dict → str
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy array / numpy scalar (np.float32, np.int64, np.bool_ ...)
        return _json_safe(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def save_result(
    side: str,
    pgnum: int,
    bn: int,
    length: int,
    features: dict,
    input_data: dict,
    recon_data: dict,
    overall_pass: bool | None = None,
    asn: int | None = None,
) -> dict:
    """
    บันทึกผลตรวจ 1 คัน / 1 ข้าง ลงตาราง prediction_break_oil_filling_machine
    (append-only → INSERT แถวใหม่ทุกครั้ง ไม่ทับของเดิม, bn ซ้ำได้)

    - side          : 'LH' หรือ 'RH'
    - pgnum, bn     : int
    - length        : จำนวนวินาทีของ sequence
    - features      : dict ผลต่อ feature  {"pressure": {"percent":..,"pass":..,...}, ...}
    - input_data    : dict ค่าจริงรายวินาที  {"lh_fillingpressure": [...], ...}
    - recon_data    : dict ค่าที่โมเดลสร้างคืน (key เหมือน input_data)
    - overall_pass  : ถ้าไม่ส่ง (None) จะคำนวณเองจาก AND ของ features[*]["pass"]
    - asn           : ยังไม่มีส่งมา → ปล่อย None ได้ (เก็บเป็น NULL)
    - predicted_at  : ใช้ default now() ของ DB

    คืน dict {"id": ..., "predicted_at": ...} ของแถวที่เพิ่งบันทึก
    """
    side = str(side).upper()
    if side not in ("LH", "RH"):
        raise ValueError(f"side ต้องเป็น 'LH' หรือ 'RH' (ได้ {side!r})")

    features = _json_safe(features)
    input_data = _json_safe(input_data)
    recon_data = _json_safe(recon_data)

    if overall_pass is None:
        # overall_pass = AND ของ pass ทุก feature
        overall_pass = all(bool(f.get("pass")) for f in features.values())

    sql = f"""
        INSERT INTO {PRED_TABLE}
            (side, pgnum, bn, asn, length, overall_pass, features, input, recon)
        VALUES
            (%(side)s, %(pgnum)s, %(bn)s, %(asn)s, %(length)s, %(overall_pass)s,
             %(features)s, %(input)s, %(recon)s)
        RETURNING id, predicted_at
    """
    params = {
        "side": side,
        "pgnum": int(pgnum),
        "bn": int(bn),
        "asn": None if asn is None else int(asn),
        "length": int(length),
        "overall_pass": bool(overall_pass),
        "features": Jsonb(features),
        "input": Jsonb(input_data),
        "recon": Jsonb(recon_data),
    }
    # pool.connection() จะ commit ให้เองเมื่อออกจาก with โดยไม่มี exception
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def save_result_from_dict(result: dict) -> dict:
    """
    รับ dict ผลลัพธ์จาก run_prediction (หลังเติม side / pgnum แล้ว) แล้วบันทึกลง DB
    key ที่ต้องมี : side, pgnum, bn, features, input, recon   (input/recon ใช้ชื่อ input_data/recon_data ก็ได้)
    key ที่ไม่บังคับ: length (ไม่มี → ใช้ความยาว array ตัวแรกของ input),
                     overall_pass (ไม่มี → คำนวณจาก features), asn
    ถ้า key ที่บังคับหายไปจะ raise KeyError บอกชื่อ key ที่ขาด
    """
    input_data = result.get("input", result.get("input_data"))
    recon_data = result.get("recon", result.get("recon_data"))
    features = result.get("features")

    missing = [name for name, v in (("input", input_data), ("recon", recon_data),
                                    ("features", features)) if v is None]
    if missing:
        raise KeyError(f"result ไม่มี key: {missing} (มี: {list(result.keys())})")

    length = result.get("length")
    if length is None:
        length = len(next(iter(input_data.values())))

    return save_result(
        side=result["side"],
        pgnum=result["pgnum"],
        bn=result["bn"],
        length=length,
        features=features,
        input_data=input_data,
        recon_data=recon_data,
        overall_pass=result.get("overall_pass"),
        asn=result.get("asn"),
    )