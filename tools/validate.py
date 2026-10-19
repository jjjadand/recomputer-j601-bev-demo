"""Exercise every retained real frame, counters, API controls and optional TCP load."""
import argparse
import base64
import hashlib
import json
import math
import pathlib
import statistics
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-test", action="store_true")
    ap.add_argument("--dataset", default="data/nuscenes-long")
    ap.add_argument("--output", default="logs/validation.json")
    ap.add_argument("--preserve-load", action="store_true", help="Validate with the currently running Ethernet benchmark")
    ap.add_argument("--expect-model", help="Require this model on every result, e.g. YOLO26m")
    ap.add_argument("--expect-resolution", type=int, help="Require the TensorRT input resolution, e.g. 960")
    ap.add_argument("--expect-semantic-bev", action="store_true", help="Require inferred road/lane masks and filled object footprints")
    ap.add_argument("--save-frames", type=int, nargs='*', default=[], help="Save BEV previews at these frame indices")
    args = ap.parse_args()
    session = requests.Session()
    session.trust_env = False
    ui = "http://127.0.0.1:8088"
    initial = session.get(ui+"/api/status", timeout=5).json()["state"]
    gateway = initial["gateway_url"]
    report = {"frames": [], "load": [], "controls": {}, "resources": [],
              "initial_controls": {k: initial[k] for k in ("playing", "fps", "index", "loop", "load_mbps", "load_direction")}}
    def control(data):
        r = session.post(ui+"/api/control", json=data, timeout=8)
        r.raise_for_status()
        return r.json()
    try:
        control({"playing": False} if args.preserve_load else {"playing": False, "load_mbps": 0})
        time.sleep(.6)
        first = session.get(ui+"/api/status", timeout=5).json()["state"]["processed"]
        time.sleep(.7)
        second = session.get(ui+"/api/status", timeout=5).json()["state"]["processed"]
        assert first == second, "Paused replay kept sending frames"
        report["controls"]["pause"] = True
        control({"step": True})
        for _ in range(20):
            third = session.get(ui+"/api/status", timeout=5).json()["state"]["processed"]
            if third == second+1:
                break
            time.sleep(.1)
        assert third == second+1, "Single-step did not produce exactly one frame"
        report["controls"]["step"] = True
        bad = session.post(ui+"/api/control", json={"fps": 0}, timeout=5)
        assert bad.status_code == 400
        report["controls"]["invalid_fps_rejected"] = True
        dataset = ROOT / args.dataset
        manifest = json.loads((dataset/"manifest.json").read_text())
        for frame in manifest["frames"]:
            packet = (dataset/frame["packet"]).read_bytes()
            assert hashlib.sha256(packet).hexdigest() == frame["sha256"]
            started = time.monotonic()
            r = session.post(gateway+"/frame", data=packet,
                             headers={"X-Frame-Index": str(frame["index"]), "X-Reset": str(int(frame["index"] == 0 or frame.get("scene_start", False)))}, timeout=60)
            r.raise_for_status()
            result = r.json()
            if frame['index'] in args.save_frames:
                preview = ROOT / pathlib.Path(args.output).with_suffix('')
                preview.with_name(preview.name+f'-frame{frame["index"]}.jpg').write_bytes(
                    base64.b64decode(result['images']['surround'].split(',')[1]))
            assert result["index"] == frame["index"] and result["timestamp"] == frame["timestamp"]
            assert result["points_count"] == frame["points"]
            assert len(result["cameras"]) == 6 and result["ground_truth_used"] is False
            if args.expect_semantic_bev:
                semantic = result['surround']
                assert semantic['render_mode'] == 'semantic_masks' and semantic['available']
                assert semantic['semantic_lane_predictions'] is True
                assert semantic['lane_source'] == 'YOLOP camera segmentation'
                assert len(semantic['pixel_counts']) == 5 and sum(semantic['pixel_counts']) == 768*768
            if args.expect_model:
                assert result.get("vision_model") == args.expect_model, "Unexpected vision model"
                assert len(result.get("gpu_stages", [])) >= 8, "GPU pipeline stages missing"
            if args.expect_resolution:
                assert result.get("vision_resolution") == args.expect_resolution, "Unexpected inference resolution"
            assert {'lidar', 'vision', 'fusion', 'map', 'surround'} <= set(result["images"])
            assert all(v.startswith("data:image/jpeg;base64,") for v in result["images"].values())
            assert len(result["pointcloud"]) > 1000
            if frame.get("scene_start", False):
                assert result["map_frames"] == 1, "Map was not reset at scene boundary"
            assert all(c["width"] == 640 and c["image"].startswith("data:image/jpeg;base64,") and len(c["projected_points"]) > 0 for c in result["cameras"])
            for detection in result["detections"]:
                assert detection["points"] >= 5 and 0.28 <= detection["score"] <= 1
                assert all(math.isfinite(x) for x in detection["center"]+detection["size"])
                assert all(x > 0 for x in detection["size"])
            for timing in result["timings"].values():
                assert math.isfinite(timing) and timing >= 0
            report["frames"].append({"index": frame["index"], "points": result["points_count"],
                                     "vision_detections": result["vision_detections"],
                                     "spatial_detections": len(result["detections"]),
                                     "timings": result["timings"],
                                     "vision_model": result.get("vision_model"),
                                     "vision_resolution": result.get("vision_resolution"),
                                     "surround": result.get("surround"),
                                     "roundtrip_ms": round((time.monotonic()-started)*1000, 2),
                                     "output_bytes": len(r.content)})
            print(f"Frame {frame['index']:02}: {result['points_count']} points, {len(result['detections'])} spatial objects, {result['timings']['total_ms']} ms", flush=True)
            if len(report["frames"]) % 20 == 0:
                telemetry = session.get(ui+"/api/status", timeout=5).json()["telemetry"]
                assert telemetry.get("connected") and not telemetry.get("telemetry_stale"), "Jetson telemetry disconnected"
                report["resources"].append({"index": frame["index"], **{k: telemetry.get(k) for k in (
                    "gpu_percent", "gpu_instant_percent", "gpu_peak_percent", "gpu_source", "gpu_window_s",
                    "cpu_percent", "memory_percent", "temperature_c", "data_link", "load_running", "load_error")}})
        if args.load_test:
            control({"playing": True, "fps": 8, "seek": 0, "load_mbps": 900, "load_direction": "bidir"})
            time.sleep(2)
            for _ in range(8):
                status = session.get(ui+"/api/status", timeout=5).json()
                t, s = status["telemetry"], status["state"]
                report["load"].append({"tx_mbps": t["tx_mbps"], "rx_mbps": t["rx_mbps"],
                                       "rx_dropped": t["rx_dropped"], "tx_dropped": t["tx_dropped"],
                                       "processed": s["processed"], "actual_fps": s["actual_fps"],
                                       "error": s["error"]})
                time.sleep(1)
            assert report["load"][-1]["processed"] > report["load"][0]["processed"]
            assert all(row["error"] is None for row in report["load"])
            assert statistics.mean(row["tx_mbps"] for row in report["load"]) > 800
            assert statistics.mean(row["rx_mbps"] for row in report["load"]) > 800
        warm = report["frames"][2:]
        report["summary"] = {"passed_frames": len(report["frames"]),
                               "pipeline_median_ms": round(statistics.median(f["timings"]["total_ms"] for f in warm), 2),
                               "vision_cuda_median_ms": round(statistics.median(f["timings"]["vision_cuda_ms"] for f in warm), 2),
                               "roundtrip_median_ms": round(statistics.median(f["roundtrip_ms"] for f in warm), 2)}
        report["passed"] = True
        (ROOT/args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report["summary"], indent=2))
    finally:
        restore = {"fps": initial["fps"], "playing": initial["playing"],
                   "seek": initial["index"], "loop": initial["loop"]}
        if not args.preserve_load or args.load_test:
            restore.update(load_mbps=initial["load_mbps"], load_direction=initial["load_direction"])
        control(restore)


if __name__ == "__main__":
    main()
