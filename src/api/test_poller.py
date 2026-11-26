"""เทสต์ poll_loop ใน main.py ด้วย DB ปลอม (ไม่ต้องมี DB จริง) — รัน (จากโฟลเดอร์ src/api): python test_poller.py

ครอบคลุมบั๊กเดิม (BN 11387): lock หลุดระหว่างที่รอบหนึ่งจบ แล้วต่อ lock กลับมา รอบนั้นต้องไม่ถูกข้าม
"""
import asyncio
import time

import main
from test_cycle import car


class FakeTable:
    """ตาราง machine ปลอม: แถวทั้งหมด + จำนวนแถวที่ "มองเห็นแล้ว" (เลื่อนเพื่อจำลองเวลาที่ผ่านไป)"""
    def __init__(self, rows, visible):
        self.rows, self.visible = rows, visible
        self.fail_next_reads = 0                       # สั่งให้การอ่านครั้งถัดไป n ครั้งพังด้วย error

    def tail(self, n):
        return self.rows[:self.visible][-n:]

    def after(self, after_id, limit):
        if self.fail_next_reads > 0:
            self.fail_next_reads -= 1
            raise RuntimeError("connection ตาย (จำลอง)")
        return [r for r in self.rows[:self.visible] if r["id"] > after_id][:limit]


class FakeLock:
    """advisory lock ปลอม: ctl['alive']=False จำลอง connection ที่ถือ lock หลุด, ctl['free']=False จำลองมีตัวอื่นถืออยู่"""
    ctl = {"alive": True, "free": True}

    def __init__(self):
        self.mine = False

    def held(self):
        return self.mine and self.ctl["alive"]

    def acquire(self):
        if self.ctl["free"]:
            self.mine, self.ctl["alive"] = True, True
            return True
        return False

    def release(self):
        self.mine = False


async def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.01)
    return False


def run_scenario(table, scenario):
    """รัน poll_loop กับ DB ปลอม แล้วให้ scenario(table, seen) ควบคุมเหตุการณ์ ; คืนรอบที่ถูกประมวลผล [(side, bn), ...]"""
    saved = (main.db.PollerLock, main.db.get_machine_tail, main.db.get_machine_rows_after, main.process_cycle,
             main.POLL_SECONDS, main.RETRY_MAX_SECONDS)
    seen = []
    FakeLock.ctl.update(alive=True, free=True)
    main.POLLER_STATUS.update(started_at=None, lock_held=False, cursor=None, last_row_at=None, last_poll_ok=None,
                              last_error=None, last_saved=None, counts={"cycles": 0, "saved": 0, "skipped": 0, "errors": 0})
    main.db.PollerLock = FakeLock
    main.db.get_machine_tail = table.tail
    main.db.get_machine_rows_after = table.after
    main.process_cycle = lambda c: seen.append((c.side, c.bn))
    main.POLL_SECONDS, main.RETRY_MAX_SECONDS = 0.01, 0.02

    async def go():
        task = asyncio.create_task(main.poll_loop())
        try:
            assert await until(lambda: main.POLLER_STATUS["cursor"] is not None), "poller ไม่เริ่มทำงาน"
            await scenario(table, seen)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(go())
    finally:
        (main.db.PollerLock, main.db.get_machine_tail, main.db.get_machine_rows_after, main.process_cycle,
         main.POLL_SECONDS, main.RETRY_MAX_SECONDS) = saved
    return seen


def three_cars():
    rows = car(99) + car(100) + car(101)               # 83 แถวต่อคัน
    return FakeTable(rows, visible=83 + 30)            # เริ่ม: คัน 99 จบแล้ว คัน 100 เดินมา 30 แถว


def test_cycle_finished_while_lock_was_lost_is_not_skipped():
    """บั๊ก BN 11387: lock หลุด -> รอบ 100 จบระหว่างนั้น -> ต่อ lock กลับมา ต้องประมวลผลรอบ 100 (ไม่ใช่ warm-up ใหม่แล้วข้าม)"""
    async def scenario(table, seen):
        assert main.poller_status()["healthy"] is True
        FakeLock.ctl.update(alive=False, free=False)           # connection ที่ถือ lock หลุด + ตัวอื่นถือแทน
        table.visible = 83 + 83 + 5                            # ระหว่างนี้คัน 100 จบ (คัน 101 เริ่มแล้ว)
        await asyncio.sleep(0.3)
        assert seen == [], f"ไม่ควรประมวลผลตอนไม่ได้ถือ lock: {seen}"
        assert main.poller_status()["healthy"] is False and main.poller_status()["lock_held"] is False
        FakeLock.ctl["free"] = True                            # lock ว่างแล้ว
        assert await until(lambda: ("LH", 100) in seen), f"รอบ 100 ถูกข้าม! seen={seen}"
        await asyncio.sleep(0.2)
        assert main.poller_status()["healthy"] is True
    seen = run_scenario(three_cars(), scenario)
    assert seen == [("LH", 100)], seen                         # ครั้งเดียวพอดี ไม่ซ้ำ


def test_transient_db_error_keeps_state_and_loses_nothing():
    async def scenario(table, seen):
        table.fail_next_reads = 5                              # อ่าน DB พัง 5 ครั้งติด
        table.visible = 83 + 83 + 5                            # ระหว่างนั้นคัน 100 จบ
        assert await until(lambda: ("LH", 100) in seen), f"รอบ 100 ถูกข้าม! seen={seen}"
        assert main.POLLER_STATUS["last_error"] and "RuntimeError" in main.POLLER_STATUS["last_error"]
        assert main.POLLER_STATUS["counts"]["errors"] >= 5
    seen = run_scenario(three_cars(), scenario)
    assert seen == [("LH", 100)], seen


def test_normal_run_processes_each_cycle_once_and_never_the_in_progress_one():
    async def scenario(table, seen):
        table.visible = 83 * 3                                 # ครบ 3 คัน (คัน 101 ยังไม่ถูกตัดจบ)
        await asyncio.sleep(0.3)
    seen = run_scenario(three_cars(), scenario)
    assert seen == [("LH", 100)], seen                         # 99 อยู่ใน warm-up (ไม่ย้อน), 101 ยังเดินอยู่


def test_db_call_times_out_instead_of_hanging():
    old = main.DB_CALL_TIMEOUT
    main.DB_CALL_TIMEOUT = 0.05
    try:
        t0 = time.monotonic()
        try:
            asyncio.run(main._db_call(time.sleep, 0.4))
            raise AssertionError("ควร TimeoutError")
        except (TimeoutError, asyncio.TimeoutError):
            pass
    finally:
        main.DB_CALL_TIMEOUT = old


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
