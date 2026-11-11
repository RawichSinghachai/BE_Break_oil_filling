"""
ws_realtime.py — FastAPI WebSocket ส่งค่า real-time ทุก 1 วินาที
ดึงค่าจาก db.get_latest_filling() (ตาราง machine) ส่ง bn / pgnum / filling_pressure / filling_volume / vacuumgun ทั้ง LH, RH

ติดตั้ง:  pip install fastapi uvicorn "psycopg[binary,pool]"
รัน:      uvicorn test1:app --host 0.0.0.0 --port 8001
ทดสอบ:    เปิดเบราว์เซอร์ http://<ip>:8001/
WebSocket: ws://<ip>:8001/ws/realtime
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import db


@asynccontextmanager
async def lifespan(app):
    db.pool.open()      # เปิด connection pool (sync) ตอนสตาร์ท
    yield
    db.pool.close()     # ปิดตอนเลิก


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/realtime")
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


# ---------- หน้าเว็บทดสอบ (เปิด http://<ip>:8001/) ----------
HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Filling Realtime</title>
<style>
 body{font-family:sans-serif;padding:24px}
 table{border-collapse:collapse;margin-top:8px}
 td,th{border:1px solid #ccc;padding:6px 16px;text-align:right}
 th{background:#f2f2f2}
 #ts{color:#555;font-size:14px}
 .st{display:inline-block;min-width:90px;text-align:center;padding:6px 16px;border-radius:4px;border:1px solid #ccc;background:#f2f2f2;color:#555}
 .st.on{background:#d4edda;border-color:#7fc08f;color:#17692e}
 .st.off{background:#eee;color:#777}
 .st.fault{background:#f8d7da;border-color:#e08a93;color:#a1202e;font-weight:bold}
 .st.ok{background:#d4edda;border-color:#7fc08f;color:#17692e}
 #common{margin-top:16px;display:flex;gap:24px;align-items:center}
</style></head><body>
<h3>Filling Real-time</h3>
<div id="ts">connecting…</div>
<table>
 <tr><th></th><th>BN</th><th>PG Num</th><th>Filling Pressure</th><th>Filling Volume</th><th>Vacuum Gun</th></tr>
 <tr><th>LH</th><td id="lh_bn">-</td><td id="lh_pg">-</td><td id="lh_p">-</td><td id="lh_v">-</td><td id="lh_g">-</td></tr>
 <tr><th>RH</th><td id="rh_bn">-</td><td id="rh_pg">-</td><td id="rh_p">-</td><td id="rh_v">-</td><td id="rh_g">-</td></tr>
</table>
<div id="common">
 <div>Master On: <span id="c_master" class="st">-</span></div>
 <div>Continue: <span id="c_continue" class="st">-</span></div>
 <div>Machine Fault: <span id="c_fault" class="st">-</span></div>
</div>
<script>
 const ws = new WebSocket(`ws://${location.host}/ws/realtime`);
 const show = v => (v === null || v === undefined) ? '-' : v;
 // แสดงสถานะ boolean/0-1 เป็นป้ายสี (ไม่มีค่า = '-')
 const setState = (id, v, txtTrue, txtFalse, clsTrue, clsFalse) => {
   const el = document.getElementById(id);
   if (v === null || v === undefined) { el.textContent = '-'; el.className = 'st'; return; }
   const on = (v === true || v === 1 || v === '1' || v === 'true');
   el.textContent = on ? txtTrue : txtFalse;
   el.className = 'st ' + (on ? clsTrue : clsFalse);
 };
 ws.onmessage = e => {
   const d = JSON.parse(e.data);
   ts.textContent = d.recorded_at || '-';
   for (const side of ['lh', 'rh']) {
     const s = d[side];
     document.getElementById(side + '_bn').textContent = show(s.bn);
     document.getElementById(side + '_pg').textContent = show(s.pgnum);
     document.getElementById(side + '_p').textContent  = show(s.filling_pressure);
     document.getElementById(side + '_v').textContent  = show(s.filling_volume);
     document.getElementById(side + '_g').textContent  = show(s.vacuumgun);
   }
   const c = d.common;
   setState('c_master',   c.master_on,     'ON',    'OFF',  'on',    'off');
   setState('c_continue', c.continue,      'ON',    'OFF',  'on',    'off');
   setState('c_fault',    c.machine_fault, 'FAULT', 'OK',   'fault', 'ok');
 };
 ws.onclose = () => ts.textContent = 'disconnected';
</script>
</body></html>
"""


@app.get("/")
async def index():
    return HTMLResponse(HTML)