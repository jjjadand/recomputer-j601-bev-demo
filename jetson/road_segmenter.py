"""GPU-resident TensorRT road and lane masks from the MIT YOLOP model."""
import pathlib

import cupy as cp
import tensorrt as trt

from gpu_images import prepare


class RoadSegmenter:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(pathlib.Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot load the road/lane TensorRT engine')
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.buffers[name] = cp.empty(self.engine.get_tensor_shape(name),
                                         dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            self.context.set_tensor_address(name, self.buffers[name].data.ptr)
        if (set(self.buffers) != {'images', 'drive_area_seg', 'lane_line_seg'}
                or self.buffers['images'].shape != (6, 3, 640, 640)
                or any(self.buffers[name].shape != (6, 2, 640, 640)
                       for name in ('drive_area_seg', 'lane_line_seg'))):
            raise RuntimeError('Road engine must have six cameras and two 640x640 segmentation heads')
        self.mean = cp.asarray([.485, .456, .406], dtype=cp.float32)[None, :, None, None]
        self.std = cp.asarray([.229, .224, .225], dtype=cp.float32)[None, :, None, None]
        self.last_gpu_ms = self.last_preprocess_ms = 0

    def infer_images(self, images):
        if len(images) != 6:
            raise ValueError('Road segmentation requires six cameras')
        with self.stream:
            start, prepared, inferred = (cp.cuda.Event() for _ in range(3))
            start.record(self.stream)
            metadata = [prepare(image, self.buffers['images'][i]) for i, image in enumerate(images)]
            self.buffers['images'] -= self.mean
            self.buffers['images'] /= self.std
            prepared.record(self.stream)
            if not self.context.execute_async_v3(self.stream.ptr):
                raise RuntimeError('Road segmentation TensorRT execution failed')
            inferred.record(self.stream)
        self.stream.synchronize()
        self.last_preprocess_ms = cp.cuda.get_elapsed_time(start, prepared)
        self.last_gpu_ms = cp.cuda.get_elapsed_time(prepared, inferred)
        return self.buffers['drive_area_seg'], self.buffers['lane_line_seg'], metadata
