"""Management gateway; inference and benchmark traffic use the configured Ethernet link."""
import asyncio
import json
import os
import pathlib
import re
import subprocess
import threading
import time

from urllib.parse import urlsplit

from telemetry import PortSampler, counter
from gpu_telemetry import sample_gpu
import psutil
from aiohttp import ClientSession, ClientTimeout, web

WORKER = os.environ.get("BEV_WORKER_URL", "http://10.77.0.2:9001")
WORKER_HOST = urlsplit(WORKER).hostname
PHYSICAL = os.environ.get("BEV_LINK_MODE") == "physical"
LINK_IFACE = os.environ.get("BEV_LINK_IFACE", "bev-host")
CAPACITY = int(os.environ.get("BEV_LINK_CAPACITY", "1000"))
MAX_LOAD = min(9500, CAPACITY*.95) if PHYSICAL else 900


class Gateway:
    def __init__(self):
        self.session = None
        self.metrics = {"mode": "physical_ethernet" if PHYSICAL else "simulated_ethernet", "capacity_mbps": CAPACITY, "interface": LINK_IFACE, "max_load_mbps": MAX_LOAD,
                        "rx_mbps": 0, "tx_mbps": 0, "rx_bytes": 0, "tx_bytes": 0,
                        "worker": {}, "gpu_percent": None, "temperature_c": None,
                        "power_w": None, "cpu_percent": 0, "memory_percent": 0}
        self.load_tasks = []
        self.load_mbps = 0
        self.load_direction = "bidir"
        self.stop = threading.Event()
        self.ports = PortSampler()
        self.load_process = None
        self.load_error = None
        self.ingress_interface = "usb0"

    def record_ingress(self, request):
        local = request.transport.get_extra_info("sockname") if request.transport else None
        if local:
            for interface, addresses in psutil.net_if_addrs().items():
                if any(address.address == local[0] for address in addresses):
                    self.ingress_interface = "usb0" if interface == "l4tbr0" else interface
                    break

    def tegrastats(self):
        process = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True)
        try:
            for line in process.stdout:
                if self.stop.is_set():
                    break
                for key, pattern, divisor in [
                    ("temperature_c", r"(?:GPU|gpu)@([\d.]+)C", 1),
                    ("power_w", r"(?:VDD_GPU_SOC|VDD_GPU|VDD_IN)\s+(\d+)mW", 1000),
                ]:
                    match = re.search(pattern, line)
                    if match:
                        self.metrics[key] = round(float(match[1])/divisor, 2)
                self.metrics["tegrastats"] = line.strip()
        finally:
            process.terminate()

    def gpu_sample(self):
        sample_gpu(self.stop, self.metrics.update)

    async def sample(self):
        previous = None
        while True:
            now = time.monotonic()
            local_ports = list(self.ports.sample())
            local_port = next((p for p in local_ports if p["interface"] == LINK_IFACE), None)
            rx = local_port["rx_bytes"] if local_port else counter(LINK_IFACE, "rx_bytes")
            tx = local_port["tx_bytes"] if local_port else counter(LINK_IFACE, "tx_bytes")
            usb_rx, usb_tx = counter("usb0", "rx_bytes"), counter("usb0", "tx_bytes")
            if previous:
                elapsed = now-previous[0]
                self.metrics.update(rx_mbps=round(max(0,rx-previous[1])*8/elapsed/1e6,2),
                                    tx_mbps=round(max(0,tx-previous[2])*8/elapsed/1e6,2),
                                    usb_rx_mbps=round(max(0,usb_rx-previous[3])*8/elapsed/1e6,2),
                                    usb_tx_mbps=round(max(0,usb_tx-previous[4])*8/elapsed/1e6,2))
            previous = (now,rx,tx,usb_rx,usb_tx)
            try:
                async with self.session.get(WORKER+"/health", timeout=ClientTimeout(total=2)) as response:
                    self.metrics["worker"] = await response.json()
                    self.metrics["connected"] = response.status == 200
                    if response.status == 200:
                        self.metrics.pop("error", None)
            except Exception as error:
                self.metrics["connected"] = False
                self.metrics["error"] = str(error)
            ports = local_ports
            peer = self.metrics.get("worker", {}).get("ethernet_port")
            if peer and all(p["interface"] != peer["interface"] for p in ports):
                ports.append(peer)
            ports.sort(key=lambda p:p["interface"])
            memory = psutil.virtual_memory()
            self.metrics.update(rx_bytes=rx,tx_bytes=tx,load_mbps=self.load_mbps,
                                load_running=bool(self.load_process and self.load_process.returncode is None) if PHYSICAL else bool(self.load_tasks),
                                load_kind="physical_tcp_benchmark" if PHYSICAL else "synthetic_tcp_load",
                                load_direction=self.load_direction,
                                load_error=self.load_error,cpu_percent=psutil.cpu_percent(),memory_percent=memory.percent,
                                memory_used_gb=round(memory.used/2**30,2),memory_total_gb=round(memory.total/2**30,2),
                                sampled_at=time.time(),ethernet_ports=ports,ingress_interface=self.ingress_interface)
            for key in ("rx_packets","tx_packets","rx_dropped","tx_dropped","rx_errors","tx_errors"):
                self.metrics[key] = counter(LINK_IFACE,key)
            physical = next((p for p in ports if p["interface"] == (LINK_IFACE if PHYSICAL else self.ingress_interface)),None)
            self.metrics["data_link"] = {
                "interface": physical["interface"] if physical else LINK_IFACE,
                "peer_interface": os.environ.get("BEV_EDGE_IFACE") if PHYSICAL else None,
                "kind": "physical_ethernet" if physical else "software_ethernet",
                "capacity_mbps": physical["speed_mbps"] or 10000 if physical else 10000,
                "hardware_capacity_mbps": 10000,
                "transport_limit_mbps": None if physical else CAPACITY,
                "carrier": physical["carrier"] if physical else True,
                "mtu": physical.get("mtu") if physical else 1500,
                "counter_source": physical.get("counter_source") if physical else "linux_interface",
                "input_mbps": self.metrics["tx_mbps"] if PHYSICAL or not physical else physical["rx_mbps"],
                "output_mbps": self.metrics["rx_mbps"] if PHYSICAL or not physical else physical["tx_mbps"],
                "input_bytes": tx if PHYSICAL or not physical else physical["rx_bytes"],
                "output_bytes": rx if PHYSICAL or not physical else physical["tx_bytes"],
            }
            await asyncio.sleep(max(.1,1-(time.monotonic()-now)))

    async def frame(self, request):
        self.record_ingress(request)
        body = await request.read()
        headers = {name: request.headers.get(name, "0") for name in ("X-Frame-Index", "X-Reset")}
        try:
            async with self.session.post(WORKER+"/frame", data=body, headers=headers,
                                         timeout=ClientTimeout(total=90)) as response:
                result = await response.read()
                return web.Response(body=result, status=response.status, content_type="application/json")
        except Exception as error:
            return web.json_response({"error": "Ethernet worker: "+str(error)}, status=502)

    async def status(self, request):
        return web.json_response(self.metrics)

    async def upload(self, rate):
        _, writer = await asyncio.open_connection(WORKER_HOST, 9010)
        chunk = bytes(256 << 10)
        start, total = time.monotonic(), 0
        try:
            while True:
                writer.write(chunk)
                await writer.drain()
                total += len(chunk)
                delay = start+total/(rate*1e6/8)-time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            writer.close()

    async def download(self, rate):
        reader, writer = await asyncio.open_connection(WORKER_HOST, 9011)
        writer.write((json.dumps({"mbps": rate})+"\n").encode())
        await writer.drain()
        try:
            while await reader.read(1 << 20):
                pass
        finally:
            writer.close()

    async def stop_load(self):
        for task in self.load_tasks:
            task.cancel()
        await asyncio.gather(*self.load_tasks,return_exceptions=True)
        self.load_tasks = []
        if self.load_process and self.load_process.returncode is None:
            self.load_process.terminate()
            try:
                await asyncio.wait_for(self.load_process.wait(),5)
            except asyncio.TimeoutError:
                self.load_process.kill()
                await self.load_process.wait()
        self.load_process = None
        self.load_mbps = 0

    async def load(self, request):
        data = await request.json()
        rate = float(data.get("mbps",0))
        direction = data.get("direction", "bidir")
        if direction not in ("bidir", "input", "output"):
            return web.json_response({"error":"Invalid benchmark direction"},status=400)
        if not 0 <= rate <= MAX_LOAD:
            return web.json_response({"error": f"Traffic rate must be between 0 and {MAX_LOAD:g} Mbps"},status=400)
        await self.stop_load()
        self.load_error = None
        self.load_direction = direction
        if rate and PHYSICAL:
            # Four independent streams in each direction. -b is per stream.
            direction_flags = ["--bidir"] if direction == "bidir" else (["-R"] if direction == "output" else [])
            # The server may still be releasing the previous direction's streams.
            for attempt in range(3):
                self.load_process = await asyncio.create_subprocess_exec(
                    "iperf3","-c",WORKER_HOST,"-B",os.environ["BEV_HOST_ADDRESS"],
                    "-p","5201",*direction_flags,"-P","4","-t","86400","-b",f"{rate/4:g}M",
                    stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                await asyncio.sleep(.3)
                if self.load_process.returncode is None:
                    break
                if attempt<2:
                    await asyncio.sleep(.4)
            if self.load_process.returncode is not None:
                self.load_error = f"Physical TCP benchmark exited ({self.load_process.returncode})"
                return web.json_response({"error":self.load_error},status=502)
        elif rate:
            self.load_tasks = []
            if direction in ("bidir", "input"):
                self.load_tasks.append(asyncio.create_task(self.upload(rate)))
            if direction in ("bidir", "output"):
                self.load_tasks.append(asyncio.create_task(self.download(rate)))
        self.load_mbps = rate
        return web.json_response({"mbps":rate,"kind":"physical_tcp_benchmark" if PHYSICAL else "synthetic_tcp_load","direction":direction})

    async def startup(self, app):
        self.session = ClientSession()
        self.sampling = asyncio.create_task(self.sample())
        threading.Thread(target=self.tegrastats, daemon=True).start()
        threading.Thread(target=self.gpu_sample, daemon=True).start()

    async def cleanup(self, app):
        self.stop.set()
        self.sampling.cancel()
        await self.stop_load()
        await asyncio.gather(self.sampling,return_exceptions=True)
        await self.session.close()


if __name__ == "__main__":
    app = web.Application(client_max_size=32 << 20)
    gateway = Gateway()
    app.router.add_post("/frame", gateway.frame)
    app.router.add_get("/status", gateway.status)
    app.router.add_post("/load", gateway.load)
    app.on_startup.append(gateway.startup)
    app.on_cleanup.append(gateway.cleanup)
    # Accept a future physical Ethernet IP while preserving USB management access.
    web.run_app(app, host=os.environ.get("BEV_GATEWAY_BIND", "0.0.0.0"), port=9000)
