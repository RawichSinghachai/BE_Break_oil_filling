import asyncio
import json

import aiomqtt

CYCLE_TOPIC = "tmt/banpho/assembly/final3/filling_equipment/BFC/BFC/cycle"


def connect_mqtt(hostname="100.124.242.18", port=1883):
    return aiomqtt.Client(hostname=hostname, port=port)


def parse_payload(raw):
    """แปลง payload เป็น dict ถ้าเป็น JSON, ถ้าไม่ใช่คืนเป็น str"""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


async def read_cycle():
    # reconnect อัตโนมัติถ้าหลุด
    while True:
        try:
            async with connect_mqtt() as client:
                await client.subscribe(CYCLE_TOPIC)
                print(f"Subscribed: {CYCLE_TOPIC}")

                async for message in client.messages:
                    data = parse_payload(message.payload)
                    print(f"[{message.topic}] {data}")
                    # TODO: ส่ง data เข้า pipeline (เช่น LSTM anomaly check) ตรงนี้

        except aiomqtt.MqttError as e:
            print(f"MQTT connection lost: {e!r} -> retry in 5s")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(read_cycle())
    except KeyboardInterrupt:
        pass