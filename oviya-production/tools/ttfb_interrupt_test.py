import asyncio
import json
import time
import websockets


WS_URL = "ws://localhost:8000/ws/conversation?user_id=test&token="  # adjust if needed


async def run():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        # send half second of silence @16k (pcm16)
        fake_chunk = (b"\x00\x00" * 8000)
        start = time.time()
        await ws.send(fake_chunk)

        ttfb_ms = None
        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "first_audio_chunk":
                ttfb_ms = (time.time() - start) * 1000.0
                print(f"TTFB: {ttfb_ms:.1f} ms")
                break

        # now send an interrupt command
        iid = "i-12345"
        await ws.send(json.dumps({"type": "interrupt", "id": iid}))

        while True:
            m = await ws.recv()
            try:
                j = json.loads(m)
            except Exception:
                continue
            if j.get("type") == "interrupt_ack" and j.get("id") == iid:
                print("interrupt ack received")
                break


if __name__ == "__main__":
    asyncio.run(run())


