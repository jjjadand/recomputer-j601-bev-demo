"""Run on the PC in the optional export environment; Jetson inference needs no Torch."""
import hashlib
import argparse
import json
import os
import pathlib

project = pathlib.Path(__file__).resolve().parent.parent
# Keep export settings in the project instead of modifying the PC's YOLO settings.
os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))
from ultralytics import YOLO

root = project / "models"
root.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolo26n', choices=['yolo26n','yolo26s','yolo26m','yolo26l'])
parser.add_argument('--imgsz', type=int, default=640)
args = parser.parse_args()
model = YOLO(root / f"{args.model}.pt")
exported = pathlib.Path(model.export(format="onnx", batch=6, imgsz=args.imgsz, dynamic=False,
                                   opset=17, simplify=False, nms=False, device="cpu"))
suffix = '' if args.model=='yolo26n' and args.imgsz==640 else f'-{args.imgsz}'
target = root / f"{args.model}-b6{suffix}.onnx"
exported.replace(target)
manifest = 'manifest.json' if not suffix else f'manifest-{args.model}{suffix}.json'
(root / manifest).write_text(json.dumps({
    "model": args.model.replace('yolo','YOLO'), "source": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{args.model}.pt",
    "license": "Ultralytics AGPL-3.0", "export": f"ONNX opset 17, six cameras, {args.imgsz}x{args.imgsz}, YOLO26 end-to-end output",
    "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "engine": "Build on target Thor: TensorRT 10.16.2 / CUDA 13.2 / FP16",
}, indent=2))
