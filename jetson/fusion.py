"""TensorRT vision + calibrated CUDA point projection and geometric BEV fusion.

This is an inspectable demo baseline, not the pretrained BEVFusion network.
No annotation/ground-truth file is consumed by the algorithm.
"""
import collections
import io
import os
import pathlib
import time

import cupy as cp
import numpy as np
import tensorrt as trt

from gpu_images import NvJpeg, Rasterizer, prepare, resize
from gpu_fusion import associate
from gpu_semantic_bev import GroundMemory, semantic_bev
from road_segmenter import RoadSegmenter

LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
          9: "traffic light", 11: "stop sign"}
COLORS = {0: (255, 191, 82), 1: (210, 148, 255), 2: (66, 236, 176), 3: (238, 143, 243),
          5: (111, 181, 255), 7: (70, 183, 255), 9: (255, 225, 100), 11: (255, 110, 110)}
CHANNELS = ["FRONT", "FRONT RIGHT", "FRONT LEFT", "BACK", "BACK LEFT", "BACK RIGHT"]
SIZE, RANGE = 512, 50.0



class Detector:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(pathlib.Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize the Thor TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.buffers = {}
        self.last_gpu_ms = 0
        self.last_preprocess_ms = 0
        self.last_postprocess_ms = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.buffers[name] = cp.empty(self.engine.get_tensor_shape(name),
                                         dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            self.context.set_tensor_address(name, self.buffers[name].data.ptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.batch_size = self.engine.get_tensor_shape(self.input_name)[0]
        self.raw_output = self.engine.get_tensor_shape(self.output_name)[1] == 84

    def infer_images(self, images):
        if self.engine.get_tensor_shape(self.output_name)[-1] != 6:
            raise RuntimeError("GPU postprocessing expects an end-to-end YOLO engine")
        self.last_gpu_ms = self.last_preprocess_ms = self.last_postprocess_ms = 0
        detections = []
        for offset in range(0, len(images), self.batch_size):
            group = images[offset:offset+self.batch_size]
            if len(group) != self.batch_size:
                raise RuntimeError("Engine batch must divide six cameras")
            with self.stream:
                start, prepared, inferred, end = (cp.cuda.Event() for _ in range(4))
                start.record(self.stream)
                metadata = [prepare(image,self.buffers[self.input_name][i]) for i,image in enumerate(group)]
                prepared.record(self.stream)
                if not self.context.execute_async_v3(self.stream.ptr):
                    raise RuntimeError("TensorRT execution failed")
                inferred.record(self.stream)
                raw = self.buffers[self.output_name]
                meta = cp.asarray(metadata,dtype=cp.float32)
                boxes = raw[:,:,:4].copy()
                boxes[:,:,0::2] = (boxes[:,:,0::2]-meta[:,None,4:5])/meta[:,None,2:3]
                boxes[:,:,1::2] = (boxes[:,:,1::2]-meta[:,None,3:4])/meta[:,None,2:3]
                boxes[:,:,0::2] = cp.clip(boxes[:,:,0::2],0,meta[:,None,1:2]-1)
                boxes[:,:,1::2] = cp.clip(boxes[:,:,1::2],0,meta[:,None,0:1]-1)
                supported = cp.isin(raw[:,:,5].astype(cp.int32),cp.asarray(list(LABELS),dtype=cp.int32))
                valid = supported & (raw[:,:,4]>=.28) & ((boxes[:,:,2]-boxes[:,:,0])>=5) & ((boxes[:,:,3]-boxes[:,:,1])>=5)
                valid &= cp.cumsum(valid,axis=1)<=45
                camera_ids = cp.broadcast_to(cp.arange(len(group),dtype=cp.float32)[:,None,None],(*raw.shape[:2],1))
                rows = cp.concatenate((boxes,raw[:,:,4:6],camera_ids),axis=2)[valid]
                end.record(self.stream)
                compact = cp.asnumpy(rows)
                self.stream.synchronize()
                self.last_preprocess_ms += cp.cuda.get_elapsed_time(start,prepared)
                self.last_gpu_ms += cp.cuda.get_elapsed_time(prepared,inferred)
                self.last_postprocess_ms += cp.cuda.get_elapsed_time(inferred,end)
            grouped = [[] for _ in group]
            for x1,y1,x2,y2,score,label,camera_id in compact:
                label=int(label)
                grouped[int(camera_id)].append({"box":[float(x1),float(y1),float(x2),float(y2)],
                    "score":round(float(score),3),"class_id":label,"label":LABELS[label]})
            detections.extend(grouped)
        return detections


class Fusion:
    def __init__(self, engine_path):
        cp.cuda.Device(0).use()
        self.vision_model = pathlib.Path(engine_path).name.split('-thor')[0].replace('yolo','YOLO')
        self.detector = Detector(engine_path)
        road_path = pathlib.Path(engine_path).parent / os.environ.get('BEV_ROAD_ENGINE', 'yolop-road-thor-b6.engine')
        if 'BEV_ROAD_ENGINE' in os.environ and not road_path.is_file():
            raise FileNotFoundError(f'Requested road/lane engine not found: {road_path}')
        self.road_segmenter = RoadSegmenter(road_path) if road_path.is_file() else None
        self.vision_resolution = self.detector.buffers[self.detector.input_name].shape[-1]
        self.history = []
        self.history_times = []
        self.map_points = None
        self.map_count = 0
        self.map_capacity = 2_000_000
        self.origin = None
        self.tracks = {}
        self.next_id = 1
        self.frame_counter = 0
        self.codec = NvJpeg()
        self.raster = Rasterizer(SIZE)
        self.surround_raster = Rasterizer(768)
        self.ground_memory = GroundMemory()

    def reset(self):
        self.ground_memory.reset()
        self.history.clear()
        self.history_times.clear()
        self.map_points = None
        self.map_count = 0
        self.origin = None
        self.tracks.clear()
        self.next_id = 1
        self.frame_counter = 0

    @staticmethod
    def occupancy(xyz):
        gx = cp.floor((RANGE - xyz[:, 1]) * SIZE / (2 * RANGE)).astype(cp.int32)
        gy = cp.floor((RANGE - xyz[:, 0]) * SIZE / (2 * RANGE)).astype(cp.int32)
        valid = (gx >= 0) & (gx < SIZE) & (gy >= 0) & (gy < SIZE) & (xyz[:, 2] > -0.6) & (xyz[:, 2] < 6)
        flat = gy[valid] * SIZE + gx[valid]
        density = cp.bincount(flat, minlength=SIZE * SIZE).reshape(SIZE, SIZE)
        height = cp.full(SIZE * SIZE, -1, dtype=cp.float32)
        cp.maximum.at(height, flat, xyz[valid, 2])
        cp.cuda.get_current_stream().synchronize()
        return density, height.reshape(SIZE, SIZE)

    @staticmethod
    def pixel(x, y):
        return int((RANGE-y)*SIZE/(2*RANGE)), int((RANGE-x)*SIZE/(2*RANGE))

    def tracking(self, detections):
        used = set()
        new = {}
        for detection in detections:
            position = np.asarray(detection["center"][:2])
            choices = [(np.linalg.norm(position - np.asarray(t["center"][:2])), tid)
                       for tid, t in self.tracks.items()
                       if tid not in used and t["class_id"] == detection["class_id"]]
            distance, tid = min(choices, default=(999, -1))
            if distance > 4:
                tid = self.next_id
                self.next_id += 1
            used.add(tid)
            detection["id"] = tid
            new[tid] = detection
        self.tracks = new

    def process(self, body, index=0, reset=False):
        started = time.perf_counter()
        if reset:
            self.reset()
        with np.load(io.BytesIO(body), allow_pickle=False) as data:
            points = np.array(data["points"], dtype=np.float32, copy=True)
            projections = np.array(data["lidar2image"], dtype=np.float32, copy=True)
            l2ego = np.array(data["lidar2ego"], dtype=np.float32, copy=True)
            l2world = np.array(data["lidar2world"], dtype=np.float32, copy=True)
            encoded_images = [np.array(data[f"cam_{i}"], copy=True) for i in range(6)]
            timestamp = int(data["timestamp"][0])
        unpack_ms = (time.perf_counter()-started)*1000
        t = time.perf_counter()
        images = self.codec.decode(encoded_images)
        jpeg_decode_ms = (time.perf_counter()-t)*1000
        if points.ndim != 2 or points.shape[1] != 5 or projections.shape != (6, 4, 4):
            raise ValueError("Expected paired nuScenes points and six calibration matrices")
        if not np.isfinite(points).all():
            raise ValueError("Corrupt sensor data")
        decode_ms = (time.perf_counter()-started)*1000
        t = time.perf_counter()
        camera_detections = self.detector.infer_images(images)
        vision_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        road_masks = self.road_segmenter.infer_images(images) if self.road_segmenter else None
        road_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        xyz = cp.asarray(points[:, :3])
        homogeneous = cp.concatenate((xyz, cp.ones((len(xyz), 1), cp.float32)), axis=1)
        ego = (homogeneous @ cp.asarray(l2ego).T)[:, :3]
        projected = [homogeneous @ cp.asarray(p).T for p in projections]
        pixels = [(p[:, 0]/cp.maximum(p[:, 2], 0.01), p[:, 1]/cp.maximum(p[:, 2], 0.01), p[:, 2]) for p in projected]
        cp.cuda.get_current_stream().synchronize()
        projection_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        density, height = self.occupancy(ego)
        bev_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        fused, semantics = associate(ego,pixels,camera_detections)
        self.tracking(fused)
        associated_points = int(cp.count_nonzero(semantics >= 0))
        cp.cuda.get_current_stream().synchronize()
        fusion_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        lidar_bev = self.raster.background(density, height)
        fused_bev = lidar_bev.copy()
        camera_bev = cp.empty_like(lidar_bev)
        camera_bev[:] = cp.asarray([22,19,13],dtype=cp.uint8)
        self.raster.boxes(fused_bev,fused,COLORS)
        self.raster.boxes(camera_bev,fused,COLORS)
        surround = None
        surround_metadata = {"available": road_masks is not None, "extent_m": 30,
                             "ground_plane_ego_z_m": 0, "max_projection_range_m": 25,
                             "render_mode": "semantic_masks", "lane_source": "YOLOP camera segmentation",
                             "semantic_lane_predictions": road_masks is not None,
                             "object_source": "LiDAR-associated geometric footprints",
                             "ground_history_max_s": self.ground_memory.max_age_s,
                             "lane_history_max_s": self.ground_memory.lane_age_s,
                             "camera_fusion": "weighted visible semantic probabilities",
                             "label_ids": {"unknown": 0, "road": 1, "lane": 2, "object": 3, "ego": 4}}
        if road_masks is not None:
            surround, labels = semantic_bev(*road_masks, projections, l2ego,
                                            camera_detections, fused, COLORS,
                                            memory=self.ground_memory, ego2world=l2world @ np.linalg.inv(l2ego),
                                            timestamp_s=timestamp/1e6)
            surround_metadata['pixel_counts'] = cp.asnumpy(cp.bincount(labels.ravel(), minlength=5)).tolist()
            surround_metadata['history_filled_pixels'] = self.ground_memory.last_filled_pixels
            surround_metadata['small_gap_filled_pixels'] = self.ground_memory.last_closed_pixels
            self.surround_raster.boxes(surround, fused, COLORS, extent=30)
        else:
            surround_metadata['reason'] = 'Road/lane TensorRT engine is not installed'
        map_started = time.perf_counter()
        if self.origin is None:
            # The accumulation map is fixed in the first frame's ego coordinates.
            self.origin = l2world @ np.linalg.inv(l2ego)
        step = max(1, len(points)//6000)
        mapped = (homogeneous[::step] @ cp.asarray(np.linalg.inv(self.origin) @ l2world, dtype=cp.float32).T)[:, :3]
        self.history.append(mapped)
        self.history_times.append(timestamp)
        if self.map_points is None:
            self.map_points = cp.empty((self.map_capacity, 3), dtype=cp.float32)
        incoming = mapped[(cp.abs(mapped[:, 0]) < 500) & (cp.abs(mapped[:, 1]) < 500) & (mapped[:, 2] > -1) & (mapped[:, 2] < 8)]
        if self.map_count + len(incoming) > self.map_capacity:
            kept = self.map_points[:self.map_count:2]
            self.map_count = len(kept)
            self.map_points[:self.map_count] = kept
        count = min(len(incoming), self.map_capacity - self.map_count)
        if count:
            self.map_points[self.map_count:self.map_count+count] = incoming[:count]
            self.map_count += count
        mapped_points = self.map_points[:self.map_count]
        map_density, map_height = self.occupancy(mapped_points)
        map_bev = self.raster.background(map_density, map_height)
        current_pose = np.linalg.inv(self.origin) @ l2world @ np.linalg.inv(l2ego)
        pose_x,pose_y = self.pixel(current_pose[0,3],current_pose[1,3])
        yy,xx=cp.ogrid[:SIZE,:SIZE]
        map_bev[(xx-pose_x)**2+(yy-pose_y)**2<=25]=cp.asarray([80,220,255],dtype=cp.uint8)
        occupied_cells = int(cp.count_nonzero(density))
        map_occupied_cells = int(cp.count_nonzero(map_density))
        cp.cuda.get_current_stream().synchronize()
        mapping_ms = (time.perf_counter()-map_started)*1000
        fusion_metadata = {'method': 'calibrated LiDAR association + occupancy overlay',
                           'feature_fusion_network': False, 'coordinate_frame': 'current ego',
                           'grid_size': SIZE, 'extent_m': RANGE,
                           'height_limits_m': [-.6, 6], 'occupied_cells': occupied_cells,
                           'associated_lidar_points': associated_points, 'object_count': len(fused)}
        mapping_metadata = {'method': 'pose-aligned rolling 3D LiDAR',
                            'display': 'interactive point cloud',
                            'pose_source': 'nuScenes supplied ego pose', 'autonomous_slam': False,
                            'coordinate_frame': 'first ego frame after reset',
                            'frames': len(self.history), 'max_frames': None,
                            'retained_points': int(self.map_count),
                            'display_extent_m': 'full trajectory',
                            'occupied_cells': map_occupied_cells,
                            'oldest_timestamp': self.history_times[0], 'newest_timestamp': timestamp,
                            'sensor_span_s': (timestamp-self.history_times[0])/1e6,
                            'ego_position_m': np.round(current_pose[:3, 3], 3).tolist()}
        sample = ego[::step]
        sample_semantics = semantics[::step]
        valid_sample = (cp.abs(sample[:,0])<60)&(cp.abs(sample[:,1])<60)&(sample[:,2]>-1)&(sample[:,2]<8)
        sample, sample_semantics = sample[valid_sample],sample_semantics[valid_sample]
        point_colors = cp.tile(cp.asarray([69,130,239],dtype=cp.int32),(len(sample),1))
        point_colors[:,1] = 130+90*cp.clip(sample[:,2]/3,0,1)
        for label,color in COLORS.items():point_colors[sample_semantics==label]=cp.asarray(color,dtype=cp.int32)
        packed_colors = point_colors[:,0]*65536+point_colors[:,1]*256+point_colors[:,2]
        point_data = cp.asnumpy(cp.column_stack((cp.round(sample.astype(cp.float64),2),packed_colors))).tolist()
        map_step = max(1, len(mapped_points)//60000)
        map_sample = mapped_points[::map_step][:60000]
        map_colors = cp.tile(cp.asarray([69,130,239],dtype=cp.int32),(len(map_sample),1))
        map_colors[:,1] = 130+90*cp.clip(map_sample[:,2]/3,0,1)
        packed_map_colors = map_colors[:,0]*65536+map_colors[:,1]*256+map_colors[:,2]
        map_point_data = cp.asnumpy(cp.column_stack((cp.round(map_sample.astype(cp.float64),2),packed_map_colors))).tolist()
        cameras = []
        thumbnails = []
        for i, image in enumerate(images):
            h, w = image.shape[:2]
            resized = resize(image,640,round(h*640/w))
            thumbnails.append(resized)
            u, v, depth = pixels[i]
            valid = (depth>1) & (depth<60) & (u>0) & (u<w) & (v>0) & (v<h)
            ids = cp.flatnonzero(valid)[::max(1, int(cp.count_nonzero(valid))//450)]
            projected_small = cp.asnumpy(cp.stack((u[ids]*640/w, v[ids]*640/w, depth[ids]), axis=1))
            ds = [{**d, "box": [round(a*640/w, 1) for a in d["box"]]} for d in camera_detections[i]]
            cameras.append({"channel": CHANNELS[i],
                            "width": resized.shape[1], "height": resized.shape[0],
                            "detections": ds, "projected_points": np.round(projected_small, 1).tolist()})
        self.frame_counter += 1
        cp.cuda.get_current_stream().synchronize()
        render_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        bev_images = [lidar_bev, camera_bev, fused_bev, map_bev] + ([surround] if surround is not None else [])
        encoded = self.codec.encode_many(thumbnails + bev_images,
                                         [82]*6 + [85]*4 + ([90] if surround is not None else []))
        for camera, image in zip(cameras, encoded):
            camera['image'] = image
        encode_ms = (time.perf_counter()-t)*1000
        return {"index": index, "timestamp": timestamp, "points_count": len(points),
                "cameras": cameras, "detections": fused,
                "vision_detections": sum(len(d) for d in camera_detections),
                "pointcloud": point_data, "map_pointcloud": map_point_data, "map_frames": len(self.history),
                "images": dict(zip(('lidar', 'vision', 'fusion', 'map', 'surround'), encoded[6:])),
                "surround": surround_metadata,
                "fusion_bev": fusion_metadata, "mapping": mapping_metadata,
                "timings": {"decode_ms": round(decode_ms, 2), "vision_ms": round(vision_ms, 2),
                            "unpack_ms": round(unpack_ms, 2), "jpeg_decode_ms": round(jpeg_decode_ms, 2),
                            "vision_cuda_ms": round(self.detector.last_gpu_ms, 2),
                            "preprocess_cuda_ms": round(self.detector.last_preprocess_ms,2),
                            "postprocess_cuda_ms": round(self.detector.last_postprocess_ms,2),
                            "road_ms": round(road_ms, 2),
                            "road_cuda_ms": round(self.road_segmenter.last_gpu_ms, 2) if self.road_segmenter else 0,
                            "road_preprocess_cuda_ms": round(self.road_segmenter.last_preprocess_ms, 2) if self.road_segmenter else 0,
                            "projection_ms": round(projection_ms, 2), "bev_ms": round(bev_ms, 2),
                            "fusion_ms": round(fusion_ms, 2), "mapping_ms": round(mapping_ms, 2),
                            "render_ms": round(render_ms, 2),
                            "jpeg_encode_ms": round(encode_ms, 2),
                            "total_ms": round((time.perf_counter()-started)*1000, 2)},
                "algorithm": "TensorRT FP16 + nvJPEG / CUDA image processing / batched LiDAR association / BEV occupancy",
                "vision_model": self.vision_model, "vision_resolution": self.vision_resolution,
                "jpeg_decode_backend": self.codec.last_decode_backend,
                "gpu_stages": [self.codec.last_decode_backend + " decode", "letterbox / RGB / normalization", "six-camera TensorRT",
                               "detection filtering", "LiDAR projection", "batched association / quantiles",
                               "BEV / map rasterization"] + (["six-camera TensorRT road / lane segmentation",
                               "calibrated semantic mask projection / object footprints"] if self.road_segmenter else []) + [
                               "point colors / downsampling", "batched JPEG encode"],
                "ground_truth_used": False,
                "detection_3d_type": "LiDAR-associated axis-aligned geometric estimate"}
