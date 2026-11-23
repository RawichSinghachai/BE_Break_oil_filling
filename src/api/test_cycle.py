"""เทสต์ของ cycle.py ด้วยข้อมูลสังเคราะห์ (ไม่ต้องใช้ DB) — รัน: python test_cycle.py"""
from datetime import datetime, timedelta, timezone

from cycle import CycleBuffer, MAX_CYCLE_LEN, side_to_lh

T0 = datetime(2026, 11, 18, 9, 0, 0, tzinfo=timezone.utc)
_counter = [0]


def row(step, bn, pg=20, asn=1, side="lh", **extra):
    """1 แถวของตาราง machine (ทั้ง LH และ RH + คอลัมน์กลาง) — ฝั่งตรงข้ามใส่ค่าอะไรก็ได้"""
    _counter[0] += 1
    r = {"id": _counter[0], "recorded_at": T0 + timedelta(seconds=1.6 * _counter[0]),
         "continue": True, "master_on": True, "machine_fault": False,
         "pumpfillingcurrent": 1.0, "pumpsupplycurrent": 2.0,
         "lh_bn": 111, "lh_pgnum": 3, "lh_asn": 9, "lh_stepnum": 1, "lh_fillingpressure": 5,
         "rh_bn": 222, "rh_pgnum": 4, "rh_asn": 8, "rh_stepnum": 1, "rh_fillingpressure": 6}
    r.update({f"{side}_stepnum": step, f"{side}_bn": bn, f"{side}_pgnum": pg, f"{side}_asn": asn})
    r.update(extra)
    return r


def car(bn, n0=8, n1=27, n2=48, **kw):
    """รถ 1 คัน: step 0 x n0, 1 x n1, 2 x n2 (รวม 83 แถว เหมือนของจริง)"""
    return [row(s, bn, **kw) for s, n in ((0, n0), (1, n1), (2, n2)) for _ in range(n)]


def feed(buf, rows):
    return [c for c in (buf.update(r) for r in rows) if c is not None]


def warmed(side="LH"):
    """buffer ที่ผ่านรอบแรก (ที่เริ่มกลางคัน) มาแล้ว: ป้อนรถ 1 คัน + ตัวตัดรอบถัดไป"""
    buf = CycleBuffer(side)
    feed(buf, car(1) + [row(0, 2)])
    buf.rows = []          # ทิ้งแถว 0 ของจุดตัดที่ค้างอยู่ เพื่อให้เทสต์ต่อไปนับแถวของรถที่ป้อนเข้าไปได้ตรง ๆ
    return buf


def test_normal_cycles_are_cut_on_step_to_zero():
    buf = CycleBuffer("LH")
    out = feed(buf, car(100) + car(101) + car(102))
    # รอบแรก (100) เริ่มก่อนเห็นจุดตัด -> ข้าม ; รอบ 101 ปิดเมื่อเจอ 0 ของ 102 ; 102 ยังไม่จบ
    assert [c.bn for c in out] == [101], [c.bn for c in out]
    assert len(out[0].rows) == 83 and not out[0].truncated
    assert out[0].rows[0]["lh_stepnum"] == 0 and out[0].rows[-1]["lh_stepnum"] == 2


def test_zero_rows_at_start_all_kept():
    out = feed(warmed(), car(5, n0=8) + [row(0, 6)])
    assert sum(1 for r in out[0].rows if r["lh_stepnum"] == 0) == 8


def test_first_cycle_started_midway_is_skipped():
    buf = CycleBuffer("LH")
    mid = [row(2, 100) for _ in range(40)]           # เริ่มกลางคัน (step 2 ล้วน)
    assert feed(buf, mid + [row(0, 101)]) == []
    assert feed(buf, car(101)[1:] + [row(0, 102)])    # รอบถัดไปปล่อยตามปกติ ... (ขาดแถว 0 แรก แต่ยังมี 0 อีก 7 แถว)


def test_stale_first_row_identity_comes_from_last_row():
    buf = warmed()
    stale = row(0, 999, pg=7, asn=77)                 # แถวแรกยังเป็น bn/pgnum/asn ของรถคันก่อน
    rest = car(200, n0=7, asn=2)
    out = feed(buf, [stale] + rest + [row(0, 201)])
    c = out[0]
    assert (c.bn, c.pgnum, c.asn) == (200, 20, 2), (c.bn, c.pgnum, c.asn)
    assert len(c.rows) == 83                          # แถวแรก (stale) ยังถูกเก็บไว้ (เหมือนระบบเดิม): 1 + 82
    assert c.rows[0]["lh_bn"] == 999


def test_machine_stop_is_capped_at_max_len():
    buf = warmed()
    stuck = car(300, n2=48) + [row(2, 300) for _ in range(5000)]    # สายหยุด: step ค้างที่ 2
    out = feed(buf, stuck)
    assert len(out) == 1 and out[0].truncated and len(out[0].rows) == MAX_CYCLE_LEN
    assert out[0].bn == 300
    assert buf.rows == []                             # ไม่เก็บแถวต่อ (ไม่โตไม่จำกัด)
    # พอสายเดินต่อ (step 0 ใหม่) ต้องเริ่มรอบใหม่ได้ปกติ และไม่ปล่อยซ้ำ
    out2 = feed(buf, car(301) + [row(0, 302)])
    assert [c.bn for c in out2] == [301] and not out2[0].truncated


def test_cycle_of_exactly_max_len_is_not_emitted_twice():
    buf = warmed()
    out = feed(buf, car(400, n0=10, n1=30, n2=MAX_CYCLE_LEN - 40) + [row(0, 401)])
    assert len(out) == 1 and len(out[0].rows) == MAX_CYCLE_LEN


def test_none_step_row_is_kept_not_dropped():
    buf = warmed()
    rows = car(500)
    rows[40]["lh_stepnum"] = None
    out = feed(buf, rows + [row(0, 501)])
    assert len(out) == 1 and len(out[0].rows) == 83


def test_incomplete_cycle_is_rejected():
    buf = warmed()
    out = feed(buf, [row(s, 600) for s, n in ((0, 8), (1, 70)) for _ in range(n)] + [row(0, 601)])
    assert out == [] and "step 2" in buf.last_reject      # หยุดกลางรอบ: ไม่ถึง step 2


def test_too_short_cycle_is_rejected():
    buf = warmed()
    out = feed(buf, [row(0, 700), row(1, 700), row(2, 700), row(0, 701)])
    assert out == [] and "สั้น" in buf.last_reject


def test_rows_are_side_filtered_and_sorted():
    buf = warmed("LH")
    rows = car(800)
    rows[10], rows[11] = rows[11], rows[10]           # สลับลำดับมาถึง
    c = feed(buf, rows + [row(0, 801)])[0]
    assert all(not k.startswith("rh_") for r in c.rows for k in r)      # ตัดฝั่งตรงข้ามทิ้ง
    assert "pumpfillingcurrent" in c.rows[0] and "machine_fault" in c.rows[0]   # คอลัมน์กลางอยู่
    ts = [r["recorded_at"] for r in c.rows]
    assert ts == sorted(ts)


def test_rh_buffer_uses_rh_columns_and_sides_are_independent():
    lh, rh = CycleBuffer("LH"), CycleBuffer("RH")
    # LH เดินปกติ ส่วน RH ค้างอยู่ step 1 ตลอด: แต่ละข้างต้องไม่ยุ่งกัน
    rows = []
    for r in car(1) + car(2) + [row(0, 3)]:
        r["rh_stepnum"] = 1
        rows.append(r)
    out_lh = feed(lh, rows)
    out_rh = feed(rh, rows)
    assert [c.bn for c in out_lh] == [2] and out_rh == []
    # ป้อน RH จริง
    rh2 = CycleBuffer("RH")
    out = feed(rh2, car(10, side="rh", pg=12) + car(11, side="rh", pg=12) + [row(0, 12, side="rh", pg=12)])
    assert [(c.side, c.bn, c.pgnum) for c in out] == [("RH", 11, 12)]
    assert all(not k.startswith("lh_") for k in out[0].rows[0])


def test_side_to_lh_renames_rh_columns_for_the_shared_model():
    r = {"id": 1, "recorded_at": 0, "machine_fault": False, "pumpfillingcurrent": 1.5,
         "rh_stepnum": 2, "rh_bn": 7, "rh_fillingpressure": 250, "rh_vacuumpump": 3}
    out = side_to_lh(r, "RH")
    assert out == {"id": 1, "recorded_at": 0, "machine_fault": False, "pumpfillingcurrent": 1.5,
                   "lh_stepnum": 2, "lh_bn": 7, "lh_fillingpressure": 250, "lh_vacuumpump": 3}, out
    # ฝั่ง RH ที่ยังมี lh_* ปนมา ต้องไม่ชนกับ rh_* ที่เปลี่ยนชื่อ
    assert side_to_lh({"lh_bn": 1, "rh_bn": 2}, "RH") == {"lh_bn": 2}
    lh = {"lh_bn": 1, "pumpsupplycurrent": 2}
    assert side_to_lh(lh, "LH") == lh


def test_rh_cycle_goes_through_side_to_lh_with_all_model_columns():
    buf = warmed("RH")
    c = feed(buf, car(30, side="rh", pg=20) + [row(0, 31, side="rh", pg=20)])[0]
    renamed = [side_to_lh(r, c.side) for r in c.rows]
    need = {"lh_stepnum", "lh_fillingpressure", "lh_bn", "pumpfillingcurrent", "pumpsupplycurrent"}
    assert need <= set(renamed[0]) and not any(k.startswith("rh_") for k in renamed[0])
    assert renamed[-1]["lh_bn"] == 30 and (c.side, c.bn, c.pgnum) == ("RH", 30, 20)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
