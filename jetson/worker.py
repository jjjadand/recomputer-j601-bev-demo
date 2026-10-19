import argparse
import asyncio
import concurrent.futures
import json
import os
import pathlib
import sys
import time
from aiohttp import web

from fusion import Fusion
from telemetry import PortSampler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from json_codec import dumps

HOST = os.environ.get("BEV_WORKER_HOST", "10.77.0.2")
EDGE_IFACE = os.environ.get("BEV_EDGE_IFACE")
PHYSICAL = os.environ.get("BEV_LINK_MODE") == "physical"


class Worker:
    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.fusion = None
        self.lock = asyncio.Lock()
        self.count = 0
        self.last_ms = 0
        self.ports = PortSampler([EDGE_IFACE] if PHYSICAL and EDGE_IFACE else [], "bev-edge")

    def process(self, body, index, reset):
        if self.fusion is None:
            models = pathlib.Path(__file__).resolve().parent.parent / "models"
            requested = os.environ.get("BEV_VISION_ENGINE")
            # Only select a larger engine explicitly after target validation.
            engine = models / (requested or "yolo26n-thor-b6.engine")
            if requested and not engine.is_file():
                raise FileNotFoundError(f"Requested vision engine not found: {engine}")
            if not engine.is_file():
                engine = models / "yolo26n-thor.engine"
            self.fusion = Fusion(engine)
        return dumps(self.fusion.process(body, index, reset))

    async def frame(self, request):
        body = await request.read()
        index = int(request.headers.get("X-Frame-Index", "0"))
        reset = request.headers.get("X-Reset", "0") == "1"
        async with self.lock:
            started = time.perf_counter()
            try:
                result = await asyncio.get_running_loop().run_in_executor(self.pool, self.process, body, index, reset)
            except Exception as error:
                import traceback
                traceback.print_exc()
                return web.json_response({"error": str(error)}, status=500)
            self.count += 1
            self.last_ms = (time.perf_counter()-started)*1000
            return web.Response(body=result, content_type='application/json')

    async def health(self, request):
        ports = self.ports.sample()
        return web.json_response({"ready": True, "model_ready": self.fusion is not None,
                                  "processed": self.count, "last_ms": round(self.last_ms, 2),
                                  "compute": "NVIDIA Thor / TensorRT FP16 / CUDA 13.2",
                                  "vision_model": self.fusion.vision_model if self.fusion else None,
                                  "vision_resolution": self.fusion.vision_resolution if self.fusion else None,
                                  "jpeg_decode_backend": self.fusion.codec.last_decode_backend if self.fusion else None,
                                  "road_segmentation_ready": bool(self.fusion and self.fusion.road_segmenter),
                                  "gpu_pipeline": "nvJPEG hardware / CUDA preprocessing / TensorRT / batched LiDAR fusion / GPU BEV rendering",
                                  "namespace": "bev-edge", "ip": HOST,
                                  "ethernet_port": ports[0] if ports else None})


async def sink(reader, writer):
    try:
        while await reader.read(1 << 20):
            pass
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def source(reader, writer):
    try:
        command = json.loads(await reader.readline())
        rate = min(9500, max(1, float(command["mbps"]))) * 1e6 / 8
        chunk = bytes(256 << 10)
        started = time.monotonic()
        total = 0
        while not reader.at_eof():
            writer.write(chunk)
            await writer.drain()
            total += len(chunk)
            delay = started+total/rate-time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
    except (ConnectionError, asyncio.CancelledError, ValueError):
        pass
    finally:
        writer.close()


async def tcp_start(app):
    app["sink"] = await asyncio.start_server(sink, HOST, 9010)
    app["source"] = await asyncio.start_server(source, HOST, 9011)
    if PHYSICAL:
        app["iperf"] = await asyncio.create_subprocess_exec(
            "iperf3", "-s", "-B", HOST, "-p", "5201",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)


async def tcp_stop(app):
    for name in ("sink", "source"):
        app[name].close()
        await app[name].wait_closed()
    app["worker"].pool.shutdown(wait=False)
    if app.get("iperf") and app["iperf"].returncode is None:
        app["iperf"].terminate()
        await app["iperf"].wait()


if __name__ == "__main__":
    app = web.Application(client_max_size=32 << 20)
    worker = Worker()
    app["worker"] = worker
    app.router.add_post("/frame", worker.frame)
    app.router.add_get("/health", worker.health)
    app.on_startup.append(tcp_start)
    app.on_cleanup.append(tcp_stop)
    web.run_app(app, host=HOST, port=9001)
