"""ค้นหาข้อมูลในตารางโดยกรองจากค่า target ที่อยู่ใน lh_bn หรือ rh_bn ด้วย psycopg (psycopg3)

ติดตั้ง:  pip install "psycopg[binary]"
รัน:     python test_db.py                # ใช้ค่า TARGET ในไฟล์
         python test_db.py ABC123         # ส่ง target ผ่าน command line
"""

import sys

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

# ---- ตั้งค่าการเชื่อมต่อ ----
DB_CONFIG = {
    "host": "100.117.187.28",
    "port": 5432,
    "user": "postgres",        # POSTGRES_USER
    "password": "1234@abcd",   # POSTGRES_PASSWORD
    "dbname": "postgres",      # POSTGRES_DB  (psycopg ใช้ dbname ไม่ใช่ database)
}

SCHEMA_NAME = "public"
TABLE_NAME = "machine"
# TABLE_NAME = "prediction_break_oil_filling_machine"

# ---- parameter สำหรับกรอง ----
TARGET = "26917"  # <-- กรอกค่าที่ต้องการค้นหาตรงนี้ (หรือส่งผ่าน argv)
LIMIT = 100     # จำกัดจำนวน row ที่ดึง (None = ไม่จำกัด)


def search_by_target(target: str, limit: int | None = LIMIT):
    """หา row ที่ lh_bn = target หรือ rh_bn = target"""
    query = sql.SQL(
        "SELECT * FROM {schema}.{table} "
        "WHERE lh_bn = %(target)s OR rh_bn = %(target)s"
    ).format(
        schema=sql.Identifier(SCHEMA_NAME),
        table=sql.Identifier(TABLE_NAME),
    )
    params = {"target": target}

    if limit is not None:
        query += sql.SQL(" LIMIT %(limit)s")
        params["limit"] = limit

    try:
        # with ... จะปิด connection ให้อัตโนมัติ
        with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        print("✅ เชื่อมต่อสำเร็จ")
        print(f"   ค้นหา target = {target!r} ใน lh_bn / rh_bn")
        print(f"   พบ {len(rows):,} rows\n")
        for i, row in enumerate(rows, start=1):
            print(f"[{i}] {row}")

        return rows

    except psycopg.errors.UndefinedTable:
        print(f"❌ ไม่พบตาราง {SCHEMA_NAME}.{TABLE_NAME} (เช็กชื่อตาราง / schema)")
    except psycopg.errors.UndefinedColumn as e:
        print(f"❌ ไม่พบ column lh_bn / rh_bn ในตาราง: {e}")
    except psycopg.OperationalError as e:
        # ต่อไม่ติด: host/port ผิด, server ไม่รัน, user/password ไม่ถูก
        print(f"❌ เชื่อมต่อไม่ได้ (OperationalError): {e}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    return None


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET
    search_by_target(target)