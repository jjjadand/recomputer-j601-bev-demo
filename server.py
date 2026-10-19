"""PC or Jetson-local replay and frontend; algorithm traffic uses the Ethernet gateway."""
import argparse
import asyncio
import collections
import json
import pathlib
import time
from urllib.parse import urlsplit
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from json_codec import dumps, loads

ROOT = pathlib.Path(__file__).resolve().parent


class Demo:
    def __init__(self, gateway, dataset, local=False):
        self.gateway = gateway.rstrip("/")
        self.dataset = pathlib.Path(dataset)
        self.manifest = json.loads((self.dataset/"manifest.json").read_text())
        self.clients = {}
        self.session = None
        self.state = {"playing": bool(local), "fps": 20 if local else 8, "index": 0, "frames": len(self.manifest["frames"]),
                      "loop": True, "load_mbps": 0, "load_direction": "bidir", "error": None, "processed": 0,
                      "actual_fps": 0, "roundtrip_ms": 0, "input_bytes": 0, "output_bytes": 0,
                      "broadcast_ms": 0,
                      "dataset": self.manifest["dataset"], "scene": self.manifest["scene"],
                      "description": self.manifest["description"], "connected": False,
                      "gateway_url": self.gateway,
                      "replay_location": "jetson" if local else "pc",
                      "native_fps": self.manifest.get("native_fps", 2),
                      "duration_s": self.manifest.get("duration_s", (len(self.manifest["frames"])-1)/2),
                      "scenes": self.manifest.get("scenes", []),
                      "frame_times": [f.get("elapsed_s", i/2) for i, f in enumerate(self.manifest["frames"])]}
        self.latest = None
        self.telemetry = {}
        self.reset = True
        self.step = False
        self.completions = collections.deque(maxlen=20)
        self.revision = 0
        self.max_load_mbps = 900

    async def broadcast(self, message):
        started = time.perf_counter()
        encoded = dumps(message).decode()
        # Each browser retains only its newest frame/status. A slow viewer must not
        # delay sensor processing or accumulate an unbounded stream of old frames.
        for pending, event in list(self.clients.values()):
            pending[message['type']] = encoded
            event.set()
        self.state['broadcast_ms'] = round((time.perf_counter()-started)*1000, 2)

    async def send_updates(self, client, pending, event):
        try:
            while not client.closed:
                await event.wait()
                event.clear()
                messages = list(pending.values())
                pending.clear()
                for encoded in messages:
                    await client.send_str(encoded)
        except (ConnectionError, RuntimeError):
            pass

    async def replay(self):
        while True:
            if not self.state["playing"] and not self.step:
                await asyncio.sleep(.05)
                continue
            self.step = False
            started = time.monotonic()
            index = self.state["index"]
            revision = self.revision
            frame = self.manifest["frames"][index]
            packet = (self.dataset/frame["packet"]).read_bytes()
            try:
                async with self.session.post(self.gateway+"/frame", data=packet,
                        headers={"X-Frame-Index": str(index),
                                 "X-Reset": str(int(self.reset or frame.get("scene_start", False)))},
                        timeout=ClientTimeout(total=90)) as response:
                    body = await response.read()
                    result = loads(body)
                    if response.status != 200:
                        raise RuntimeError(result.get("error", f"HTTP {response.status}"))
                if revision == self.revision:
                    self.reset = False
                now = time.monotonic()
                self.completions.append(now)
                actual_fps = (len(self.completions)-1)/(now-self.completions[0]) if len(self.completions)>1 else 0
                self.state.update(connected=True, error=None, processed=self.state["processed"]+1,
                                  actual_fps=round(actual_fps, 2), roundtrip_ms=round((now-started)*1000, 2),
                                  input_bytes=len(packet), output_bytes=len(body), last_received=time.time(),
                                  displayed_index=index, scene=frame.get("scene", self.manifest["scene"]),
                                  elapsed_s=frame.get("elapsed_s", index/self.state["native_fps"]),
                                  scene_index=frame.get("scene_index", index))
                scene = next((s for s in self.state["scenes"] if s["name"] == self.state["scene"]), None)
                if scene:
                    self.state["description"] = scene["description"]
                self.latest = result
                await self.broadcast({"type": "frame", "result": result, "state": self.state})
                # A seek command received during inference must win over automatic advancement.
                if revision == self.revision and self.state["index"] == index:
                    if index+1 >= self.state["frames"]:
                        if self.state["loop"]:
                            self.state["index"] = 0
                            self.reset = True
                        else:
                            self.state["playing"] = False
                    else:
                        self.state["index"] = index+1
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.state.update(connected=False, error=str(error))
                await self.broadcast({"type": "status", "state": self.state, "telemetry": self.telemetry})
                await asyncio.sleep(2)
            await asyncio.sleep(max(0, 1/self.state["fps"]-(time.monotonic()-started)))

    async def sample(self):
        while True:
            try:
                async with self.session.get(self.gateway+"/status", timeout=ClientTimeout(total=3)) as response:
                    self.telemetry = await response.json()
                    self.max_load_mbps = float(self.telemetry.get("max_load_mbps", self.max_load_mbps))
            except Exception as error:
                self.telemetry = {**self.telemetry, "connected": False, "error": str(error), "telemetry_stale": True}
            await self.broadcast({"type": "status", "state": self.state, "telemetry": self.telemetry})
            await asyncio.sleep(1)

    async def status(self, request):
        return web.json_response({"state": self.state, "telemetry": self.telemetry,
                                  "dataset": self.manifest})

    async def control(self, request):
        data = await request.json()
        try:
            if "gateway" in data:
                url = str(data["gateway"]).strip().rstrip("/")
                parsed = urlsplit(url)
                if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                        parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
                    raise ValueError("Use an HTTP(S) gateway address with host and optional port")
                if parsed.port is not None and not 1 <= parsed.port <= 65535:
                    raise ValueError("Invalid gateway port")
                if url != self.gateway:
                    self.gateway = url
                    self.state.update(gateway_url=url, connected=False, error=None)
                    self.reset = True
                    self.revision += 1
            if "fps" in data:
                fps = float(data["fps"])
                if not 1 <= fps <= 20:
                    raise ValueError("FPS must be between 1 and 20")
                self.state["fps"] = fps
                self.completions.clear()
            if "playing" in data:
                self.state["playing"] = bool(data["playing"])
                self.completions.clear()
                if not self.state["playing"]:
                    self.state["actual_fps"] = 0
            if "loop" in data:
                self.state["loop"] = bool(data["loop"])
            if "seek" in data:
                index = int(data["seek"])
                if not 0 <= index < self.state["frames"]:
                    raise ValueError("Frame index out of range")
                self.state["index"] = index
                self.revision += 1
                self.reset = True
                self.completions.clear()
            if "scene" in data:
                scene = next((s for s in self.state["scenes"] if s["name"] == data["scene"]), None)
                if scene is None:
                    raise ValueError("Unknown scene")
                self.state["index"] = scene["start_index"]
                self.state["scene"] = scene["name"]
                self.revision += 1
                self.reset = True
                self.step = not self.state["playing"]
                self.completions.clear()
            if data.get("step"):
                self.state["playing"] = False
                self.step = True
            if "load_mbps" in data or "load_direction" in data:
                rate = float(data.get("load_mbps", self.state["load_mbps"]))
                direction = data.get("load_direction", self.state["load_direction"])
                if direction not in ("bidir", "input", "output"):
                    raise ValueError("Invalid benchmark direction")
                limit = self.max_load_mbps
                if not 0 <= rate <= limit:
                    raise ValueError(f"Link load must be between 0 and {limit:g} Mbps")
                async with self.session.post(self.gateway+"/load", json={"mbps": rate, "direction": direction},
                                             timeout=ClientTimeout(total=5)) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise RuntimeError(result.get("error", "Traffic control failed"))
                self.state["load_mbps"] = rate
                self.state["load_direction"] = direction
            return web.json_response(self.state)
        except (ValueError, RuntimeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=2 << 20)
        await ws.prepare(request)
        pending, event = {}, asyncio.Event()
        self.clients[ws] = pending, event
        await ws.send_json({"type": "status", "state": self.state, "telemetry": self.telemetry})
        if self.latest is not None:
            await ws.send_json({"type": "frame", "result": self.latest, "state": self.state})
        sender = asyncio.create_task(self.send_updates(ws, pending, event))
        try:
            async for message in ws:
                if message.type == WSMsgType.ERROR:
                    break
        finally:
            self.clients.pop(ws, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        return ws

    async def startup(self, app):
        self.session = ClientSession()
        self.tasks = [asyncio.create_task(self.replay()), asyncio.create_task(self.sample())]

    async def cleanup(self, app):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            await self.session.post(self.gateway+"/load", json={"mbps": 0}, timeout=ClientTimeout(total=2))
        except Exception:
            pass
        await self.session.close()
        for ws in list(self.clients):
            await ws.close()


def application(gateway, dataset, local=False):
    app = web.Application()
    demo = Demo(gateway, dataset, local)
    app["demo"] = demo
    app.router.add_get("/api/status", demo.status)
    app.router.add_post("/api/control", demo.control)
    app.router.add_get("/ws", demo.websocket)
    app.router.add_get("/", lambda request: web.FileResponse(ROOT/"frontend/index.html"))
    app.router.add_static("/assets/", ROOT/"frontend")
    app.router.add_static("/vendor/", ROOT/"node_modules/three/build")
    app.on_startup.append(demo.startup)
    app.on_cleanup.append(demo.cleanup)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway")
    parser.add_argument("--local", action="store_true", help="Replay and view on the Jetson; use its loopback gateway")
    parser.add_argument("--dataset", default=str(ROOT/"data/nuscenes-long"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    gateway = args.gateway or ("http://127.0.0.1:9000" if args.local else "http://192.168.55.18:9000")
    web.run_app(application(gateway, args.dataset, args.local), host=args.host, port=args.port)
