"""Retain calibrated, timestamp-paired sensor sequences from the official mini archive.

The archive is streamed, only selected sensor files and required calibration metadata
are kept. Predictions never use dataset annotations.
"""
import argparse
import bisect
import hashlib
import json
import pathlib
import tarfile
import time

import numpy as np
import requests

URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
CHANNELS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
METADATA = {"sensor", "calibrated_sensor", "ego_pose", "sample", "scene", "sample_data"}


def transform(rotation, translation):
    w, x, y, z = np.asarray(rotation) / np.linalg.norm(rotation)
    r = [[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
         [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
         [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = translation
    return out


def select_frames(metadata, scene_count, frame_limit, sample_hz):
    sensors = {x["token"]: x for x in metadata["sensor"]}
    calibrations = {x["token"]: x for x in metadata["calibrated_sensor"]}
    poses = {x["token"]: x for x in metadata["ego_pose"]}
    samples = {x["token"]: x for x in metadata["sample"]}
    scenes = sorted(metadata["scene"], key=lambda s: (s["name"] != "scene-0061", s["name"]))[:scene_count]
    scene_ids = {s["token"] for s in scenes}
    streams = {s["token"]: {c: [] for c in CHANNELS + ["LIDAR_TOP"]} for s in scenes}
    for data in metadata["sample_data"]:
        scene_id = samples[data["sample_token"]]["scene_token"]
        if scene_id not in scene_ids:
            continue
        channel = sensors[calibrations[data["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if channel in streams[scene_id]:
            streams[scene_id][channel].append(data)
    chosen = []
    for scene in scenes:
        stream = streams[scene["token"]]
        for records in stream.values():
            records.sort(key=lambda d: d["timestamp"])
        camera_times = {c: [d["timestamp"] for d in stream[c]] for c in CHANNELS}
        last_timestamp = None
        first_key = samples[scene["first_sample_token"]]["timestamp"]
        last_key = samples[scene["last_sample_token"]]["timestamp"]
        scene_index = 0
        for lidar in stream["LIDAR_TOP"]:
            timestamp = lidar["timestamp"]
            if not first_key-50000 <= timestamp <= last_key+50000:
                continue
            if sample_hz == 0 and not lidar["is_key_frame"]:
                continue
            # Sensors have slight clock jitter; select approximately 10 Hz from 20 Hz LiDAR.
            if sample_hz and last_timestamp is not None and timestamp-last_timestamp < 1e6/sample_hz-12000:
                continue
            paired = {"LIDAR_TOP": lidar}
            for channel in CHANNELS:
                times = camera_times[channel]
                pos = bisect.bisect_left(times, timestamp)
                candidates = [i for i in (pos-1, pos) if 0 <= i < len(times)]
                if not candidates:
                    break
                nearest = min(candidates, key=lambda i: abs(times[i]-timestamp))
                if abs(times[nearest]-timestamp) > 80000:
                    break
                paired[channel] = stream[channel][nearest]
            if len(paired) != 7:
                continue
            chosen.append({"scene": scene, "scene_index": scene_index, "timestamp": timestamp,
                           "token": lidar["token"], "data": paired})
            scene_index += 1
            last_timestamp = timestamp
            if len(chosen) >= frame_limit:
                break
        print(f"Selected {scene['name']}: {scene_index} paired frames", flush=True)
        if len(chosen) >= frame_limit:
            break
    if not chosen:
        raise RuntimeError("No paired frames found")
    return chosen, calibrations, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes-long")
    ap.add_argument("--frames", type=int, default=600, help="Maximum retained frames across scenes")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--sample-hz", type=float, default=10, help="0 for original keyframes only")
    args = ap.parse_args()
    if args.frames < 1 or args.scenes < 1 or not 0 <= args.sample_hz <= 20:
        ap.error("Frames/scenes must be positive; sample rate must be 0–20 Hz")
    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    metadata, wanted, chosen = {}, None, []
    started = time.monotonic()
    response = requests.get(URL, stream=True, timeout=(15, 90))
    response.raise_for_status()
    response.raw.decode_content = False
    retained, next_progress = 0, time.monotonic()+15
    with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.startswith("v1.0-mini/") and member.name.endswith(".json"):
                key = pathlib.PurePosixPath(member.name).stem
                if key in METADATA:
                    metadata[key] = json.load(archive.extractfile(member))
                continue
            if wanted is None and member.name.startswith("samples/"):
                chosen, calibrations, poses = select_frames(metadata, args.scenes, args.frames, args.sample_hz)
                wanted = {d["filename"] for f in chosen for d in f["data"].values()}
                print(f"Total {len(chosen)} frames / {len(wanted)} unique sensor files", flush=True)
            if wanted and member.name in wanted:
                relative = pathlib.PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe archive path")
                path = root / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                contents = archive.extractfile(member).read()
                path.write_bytes(contents)
                retained += len(contents)
                wanted.remove(member.name)
                if not wanted:
                    break
            if time.monotonic() >= next_progress:
                print(f"Remaining {len(wanted or [])} files; retained {retained/1e6:.1f} MB; "
                      f"elapsed {time.monotonic()-started:.0f} s", flush=True)
                next_progress = time.monotonic()+15
    response.close()
    if wanted:
        raise RuntimeError(f"Archive ended before {len(wanted)} selected files were found")
    output, scene_manifest = [], []
    elapsed, scene_start_time = 0., None
    for index, frame in enumerate(chosen):
        data, scene = frame["data"], frame["scene"]
        if frame["scene_index"] == 0:
            if output:
                elapsed = output[-1]["elapsed_s"] + 1/(args.sample_hz or 2)
            scene_start_time = frame["timestamp"]
            scene_manifest.append({"name": scene["name"], "description": scene["description"],
                                   "start_index": index, "frames": 0})
        scene_manifest[-1]["frames"] += 1
        lidar = data["LIDAR_TOP"]
        lc, lp = calibrations[lidar["calibrated_sensor_token"]], poses[lidar["ego_pose_token"]]
        l2ego = transform(lc["rotation"], lc["translation"])
        l2world = transform(lp["rotation"], lp["translation"]) @ l2ego
        projections, arrays = [], {}
        for i, channel in enumerate(CHANNELS):
            cam = data[channel]
            cc, cp = calibrations[cam["calibrated_sensor_token"]], poses[cam["ego_pose_token"]]
            c2world = transform(cp["rotation"], cp["translation"]) @ transform(cc["rotation"], cc["translation"])
            intrinsic = np.eye(4)
            intrinsic[:3, :3] = cc["camera_intrinsic"]
            projections.append(intrinsic @ np.linalg.inv(c2world) @ l2world)
            arrays[f"cam_{i}"] = np.frombuffer((root / cam["filename"]).read_bytes(), dtype=np.uint8)
        arrays["points"] = np.fromfile(root / lidar["filename"], dtype="<f4").reshape(-1, 5)
        arrays["lidar2image"] = np.asarray(projections, dtype=np.float32)
        arrays["lidar2ego"] = l2ego.astype(np.float32)
        arrays["lidar2world"] = l2world.astype(np.float32)
        arrays["timestamp"] = np.asarray([frame["timestamp"]], dtype=np.int64)
        packet = root / f"frame-{index:04d}.npz"
        np.savez(packet, **arrays)
        output.append({"index": index, "token": frame["token"], "timestamp": frame["timestamp"],
                       "scene": scene["name"], "scene_index": frame["scene_index"],
                       "scene_start": frame["scene_index"] == 0,
                       "elapsed_s": round(elapsed+(frame["timestamp"]-scene_start_time)/1e6, 4),
                       "scene_elapsed_s": round((frame["timestamp"]-scene_start_time)/1e6, 4),
                       "camera_timestamps": [data[c]["timestamp"] for c in CHANNELS],
                       "packet": packet.name, "bytes": packet.stat().st_size,
                       "points": len(arrays["points"]),
                       "sha256": hashlib.sha256(packet.read_bytes()).hexdigest()})
        if index % 100 == 0:
            print(f"Packed {index+1}/{len(chosen)} frames", flush=True)
    manifest = {"dataset": "nuScenes v1.0-mini", "scene": scene_manifest[0]["name"],
                "description": scene_manifest[0]["description"], "source": URL,
                "license": "nuScenes CC BY-NC-SA 4.0; research/demo use",
                "channels": CHANNELS, "native_fps": args.sample_hz or 2,
                "duration_s": output[-1]["elapsed_s"], "scenes": scene_manifest,
                "frames": output, "retained_sensor_bytes": retained,
                "ground_truth_used_for_inference": False}
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(root / "manifest.json")
    print(f"Ready: {len(output)} paired packets, {len(scene_manifest)} scenes, "
          f"{retained/1e6:.1f} MB sensors, {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
