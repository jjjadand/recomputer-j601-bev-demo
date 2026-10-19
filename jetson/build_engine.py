import argparse
import pathlib
import time
import tensorrt as trt

root = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/yolo26n.onnx")
ap.add_argument("--output", default="models/yolo26n-thor.engine")
ap.add_argument("--aux-streams", type=int, default=2)
ap.add_argument("--optimization-level", type=int, choices=range(6), default=3)
args = ap.parse_args()
model = root / args.model
target = root / args.output
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse(model.read_bytes()):
    raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)
config.max_aux_streams = args.aux_streams
config.builder_optimization_level = args.optimization_level
started = time.monotonic()
engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError("TensorRT engine build failed")
target.write_bytes(bytes(engine))
print(f"Built {target}: {target.stat().st_size/1e6:.1f} MB in {time.monotonic()-started:.1f} s", flush=True)
