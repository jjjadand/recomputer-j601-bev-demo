"""Continuous NVML sampling for the short GPU bursts in the fusion pipeline."""
import collections
import ctypes
import math
import time


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlReader:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        signatures = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetUtilizationRates": [ctypes.c_void_p, ctypes.POINTER(Utilization)],
        }
        for name, argtypes in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        self.check(self.lib.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        try:
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def read(self):
        value = Utilization()
        self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(value)))
        return value.gpu

    def close(self):
        self.lib.nvmlShutdown()


class GpuWindow:
    def __init__(self, seconds=2.0):
        self.seconds = seconds
        self.samples = collections.deque()

    def add(self, now, value):
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("GPU utilization must be between 0 and 100")
        self.samples.append((now, value))
        while self.samples and self.samples[0][0] <= now-self.seconds:
            self.samples.popleft()
        values = [sample[1] for sample in self.samples]
        return {"gpu_percent": round(sum(values)/len(values), 1),
                "gpu_instant_percent": value, "gpu_peak_percent": max(values),
                "gpu_window_s": self.seconds, "gpu_samples": len(values)}


def sample_gpu(stop, publish, interval=.1):
    """Use one NVML session. Publish a real rolling mean including zero samples."""
    while not stop.is_set():
        reader = None
        try:
            reader = NvmlReader()
            window = GpuWindow()
            while not stop.is_set():
                started = time.monotonic()
                metrics = window.add(started, reader.read())
                metrics.update(gpu_source="nvml", gpu_sample_interval_ms=round(interval*1000),
                               gpu_sampled_at=time.time(), gpu_error=None)
                publish(metrics)
                stop.wait(max(0, interval-(time.monotonic()-started)))
        except (OSError, AttributeError, RuntimeError, ValueError) as error:
            publish({"gpu_percent": None, "gpu_instant_percent": None,
                     "gpu_peak_percent": None, "gpu_samples": 0,
                     "gpu_error": str(error), "gpu_source": "unavailable"})
            stop.wait(2)
        finally:
            if reader is not None:
                reader.close()
