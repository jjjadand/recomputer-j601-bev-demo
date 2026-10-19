import threading
import unittest
from unittest.mock import Mock, patch

from jetson.gpu_telemetry import GpuWindow, sample_gpu


class GpuTelemetryTests(unittest.TestCase):
    def test_real_idle_samples_are_included_in_average(self):
        window = GpuWindow()
        window.add(0, 40)
        metrics = window.add(.1, 0)
        self.assertEqual(metrics["gpu_percent"], 20)
        self.assertEqual(metrics["gpu_instant_percent"], 0)
        self.assertEqual(metrics["gpu_peak_percent"], 40)

    def test_idle_window_expires_previous_work(self):
        window = GpuWindow()
        window.add(0, 40)
        window.add(1.9, 0)
        metrics = window.add(2, 0)
        self.assertEqual(metrics["gpu_percent"], 0)
        self.assertEqual(metrics["gpu_peak_percent"], 0)

    def test_read_failure_clears_previous_utilization(self):
        reader = Mock()
        reader.read.side_effect = [40, RuntimeError("GPU unavailable")]
        stop, published = threading.Event(), []

        def publish(metrics):
            published.append(metrics)
            if metrics.get("gpu_error"):
                stop.set()

        with patch("jetson.gpu_telemetry.NvmlReader", return_value=reader):
            sample_gpu(stop, publish, interval=0)
        self.assertEqual(published[0]["gpu_percent"], 40)
        self.assertIsNone(published[-1]["gpu_percent"])
        self.assertEqual(published[-1]["gpu_source"], "unavailable")
        reader.close.assert_called_once()

    def test_missing_nvml_is_unknown_instead_of_zero(self):
        stop, published = threading.Event(), []

        def publish(metrics):
            published.append(metrics)
            stop.set()

        with patch("jetson.gpu_telemetry.NvmlReader", side_effect=OSError("No NVML")):
            sample_gpu(stop, publish)
        self.assertIsNone(published[0]["gpu_percent"])
        self.assertIsNone(published[0]["gpu_instant_percent"])
        self.assertEqual(published[0]["gpu_error"], "No NVML")


if __name__ == "__main__":
    unittest.main()
