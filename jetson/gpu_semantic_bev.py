"""Project semantic probabilities, not camera textures, onto the calibrated ground.

Fuse visible camera probabilities and remember pose-aligned ground observations.
Objects are filled geometric footprints; objects are never accumulated.
Labels: 0 unknown, 1 road, 2 lane marking, 3 object, 4 ego vehicle.
"""
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import binary_closing

PALETTE = np.array([[22, 19, 13], [96, 77, 54], [130, 237, 250],
                    [160, 210, 60], [215, 225, 215]], dtype=np.uint8)  # BGR

_PROJECT = cp.RawKernel(r'''
extern "C" __global__ void semantic(const float* road,const float* lane,
 const float* matrices,const float* meta,const float* boxes,int nboxes,
 unsigned char* labels,unsigned char* visibility,int size,int mask_size,float extent,float max_range){
 int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=size*size)return;
 float x=extent-(pixel/size+.5f)*2*extent/size;
 float y=extent-(pixel%size+.5f)*2*extent/size;
 unsigned char label=0;float total=0.f,road_sum=0.f,lane_sum=0.f,lane_weight=0.f;
 if(x*x+y*y<=max_range*max_range){
  for(int camera=0;camera<6;camera++){
   const float* p=matrices+camera*16;const float* m=meta+camera*5;
   float depth=p[8]*x+p[9]*y+p[11];if(depth<=1.f||depth>max_range)continue;
   float u=(p[0]*x+p[1]*y+p[3])/depth,v=(p[4]*x+p[5]*y+p[7])/depth;
   float side=(u-m[1]*.5f)/(m[1]*.5f);
   if(fabsf(side)>.98f||v<2||v>=m[0]-2)continue;
   // Edge views contribute smoothly instead of cutting an artificial seam.
   float central=fmaxf(.02f,1-side*side);
   float weight=central*central*central*central/depth;
   bool blocked=false;
   for(int b=0;b<nboxes;b++){
    const float* box=boxes+b*5;
    if((int)box[0]==camera&&u>=box[1]&&u<=box[3]&&v>=box[2]&&v<=box[4]){blocked=true;break;}
   }
   if(blocked)continue;
   int col=(int)floorf(u*m[2]+m[4]),row=(int)floorf(v*m[2]+m[3]);
   if(col<0||col>=mask_size||row<0||row>=mask_size)continue;
   int area=mask_size*mask_size,offset=camera*2*area+row*mask_size+col;
   float rp=road[offset+area]/fmaxf(1e-6f,road[offset]+road[offset+area]);
   float lp=lane[offset+area]/fmaxf(1e-6f,lane[offset]+lane[offset+area]);
   total+=weight;road_sum+=rp*weight;
   // Only road-supported lane evidence participates in lane fusion.
   if(rp>=.20f){lane_sum+=lp*weight;lane_weight+=weight;}
  }
 }
 float rp=total>0?road_sum/total:0.f;
 if(total>0&&rp>=.55f)label=1;
 if(lane_weight>0&&lane_sum/lane_weight>=.60f)label=2;
 // 0: unobserved/occluded; 1: visible; 2: visible strong non-road evidence.
 visibility[pixel]=total<=0?0:rp<.15f?2:1;
 labels[pixel]=label;
}
''', 'semantic')

_WARP = cp.RawKernel(r'''
extern "C" __global__ void warp_ground(const unsigned char* previous,const float* ages,
 const float* transform,unsigned char* warped,float* warped_age,
 int size,float extent,float dt,float max_age){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;
 float x=extent-(i/size+.5f)*2*extent/size,y=extent-(i%size+.5f)*2*extent/size;
 float px=transform[0]*x+transform[1]*y+transform[3];
 float py=transform[4]*x+transform[5]*y+transform[7];
 int row=(int)floorf((extent-px)*size/(2*extent));
 int col=(int)floorf((extent-py)*size/(2*extent));
 warped[i]=0;warped_age[i]=max_age+1;
 if(row<0||row>=size||col<0||col>=size)return;
 int j=row*size+col;float age=ages[j]+dt;
 if(age<=max_age&&(previous[j]==1||previous[j]==2)){
  warped[i]=previous[j];warped_age[i]=age;
 }
}
''', 'warp_ground')


class GroundMemory:
    """Bounded observation history, aligned in ego coordinates using dataset poses."""
    def __init__(self, max_age_s=6., lane_age_s=.3, hole_width_m=.4):
        self.max_age_s, self.lane_age_s = max_age_s, lane_age_s
        self.hole_width_m = hole_width_m
        self.reset()

    def reset(self):
        self.labels = self.ages = self.pose = self.timestamp = self.geometry = None
        self.last_filled_pixels = self.last_closed_pixels = 0

    def update(self, labels, visibility, pose, timestamp, extent, max_range):
        size = labels.shape[0]
        pose = np.asarray(pose, dtype=np.float64)
        timestamp = float(timestamp)
        geometry = (size, extent, max_range)
        dt = timestamp-self.timestamp if self.timestamp is not None else 0
        if (self.geometry != geometry or dt <= 0 or dt > 1
                or (self.pose is not None and np.linalg.norm(pose[:3, 3]-self.pose[:3, 3]) > 6)):
            self.reset()
        # Close only small gaps within visible road support. Lane dashes remain
        # separate and unobserved/occluded pixels are excluded from interpolation.
        radius = int(self.hole_width_m*size/(2*extent)/2)
        road = (labels == 1) | (labels == 2)
        if radius:
            closed = binary_closing(road, structure=cp.ones((2*radius+1, 2*radius+1), dtype=cp.bool_))
            added = closed & ~road & (visibility == 1)
            labels = cp.where(added, 1, labels).astype(cp.uint8)
            self.last_closed_pixels = int(cp.count_nonzero(added))
        age = cp.zeros(labels.shape, dtype=cp.float32)
        self.last_filled_pixels = 0
        if self.labels is not None:
            warped = cp.empty_like(labels)
            warped_age = cp.empty_like(age)
            transform = cp.asarray(np.linalg.inv(self.pose) @ pose, dtype=cp.float32)
            _WARP(((size*size+255)//256,), (256,),
                  (self.labels, self.ages, transform, warped, warped_age,
                   np.int32(size), np.float32(extent), np.float32(dt), np.float32(self.max_age_s)))
            coordinates = extent-(cp.arange(size, dtype=cp.float32)+.5)*2*extent/size
            within_range = coordinates[:, None]**2+coordinates[None, :]**2 <= max_range**2
            # Keep the observed ground beneath an old marking, while removing
            # its lane color quickly so reprojection noise cannot grow a stripe.
            warped = cp.where((warped == 2) & (warped_age > self.lane_age_s), 1, warped).astype(cp.uint8)
            support = (warped == 1) | (warped == 2)
            added = (labels == 0) & (visibility != 2) & within_range & support
            labels = cp.where(added, warped, labels).astype(cp.uint8)
            age = cp.where(added, warped_age, age)
            self.last_filled_pixels = int(cp.count_nonzero(added))
        self.labels, self.ages = labels.copy(), age
        self.pose, self.timestamp, self.geometry = pose.copy(), timestamp, geometry
        return labels

_OBJECTS = cp.RawKernel(r'''
extern "C" __global__ void footprints(unsigned char* labels,unsigned char* image,
 const float* objects,int count,int size,float extent){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;
 float x=extent-(i/size+.5f)*2*extent/size,y=extent-(i%size+.5f)*2*extent/size;
 for(int d=0;d<count;d++){
  const float* b=objects+d*7;
  if(fabsf(x-b[0])<=b[2]*.5f&&fabsf(y-b[1])<=b[3]*.5f){
   labels[i]=3;
   for(int c=0;c<3;c++)image[i*3+c]=(unsigned char)b[4+c];
   break;
  }
 }
 if(fabsf(x)<2.3f&&fabsf(y)<.95f){
  labels[i]=4;bool glass=(x>.6f&&x<1.5f)||(x<-1.f&&x>-1.65f);
  image[i*3]=glass?65:215;image[i*3+1]=glass?80:225;image[i*3+2]=glass?65:215;
 }
}
''', 'footprints')


def semantic_bev(road, lane, metadata, lidar2image, lidar2ego,
                 camera_detections, detections, colors, size=768, extent=30., max_range=25.,
                 memory=None, ego2world=None, timestamp_s=None):
    road, lane = cp.ascontiguousarray(road, dtype=cp.float32), cp.ascontiguousarray(lane, dtype=cp.float32)
    if (road.shape != lane.shape or road.ndim != 4 or road.shape[:2] != (6, 2)
            or road.shape[2] != road.shape[3] or len(metadata) != 6
            or len(camera_detections) != 6 or size <= 0 or extent <= 0 or max_range <= 0):
        raise ValueError('Expected paired six-camera probabilities, letterbox metadata and a positive BEV extent')
    matrices = cp.asarray(np.asarray(lidar2image) @ np.linalg.inv(lidar2ego), dtype=cp.float32)
    boxes = [[i, *d['box']] for i, group in enumerate(camera_detections) for d in group
             if d['class_id'] in (0, 1, 2, 3, 5, 7)]
    labels = cp.empty((size, size), dtype=cp.uint8)
    visibility = cp.empty_like(labels)
    _PROJECT(((size*size+255)//256,), (256,),
             (road, lane, matrices, cp.asarray(metadata, dtype=cp.float32),
              cp.asarray(boxes, dtype=cp.float32), np.int32(len(boxes)), labels, visibility,
              np.int32(size), np.int32(road.shape[-1]), np.float32(extent), np.float32(max_range)))
    if memory is not None:
        if ego2world is None or timestamp_s is None:
            raise ValueError('Ground memory requires a pose and sensor timestamp')
        labels = memory.update(labels, visibility, ego2world, timestamp_s, extent, max_range)
    image = cp.asarray(PALETTE)[labels]
    objects = [[*d['center'][:2], *d['size'][:2], *colors[d['class_id']][::-1]] for d in detections]
    _OBJECTS(((size*size+255)//256,), (256,),
             (labels, image, cp.asarray(objects, dtype=cp.float32), np.int32(len(objects)),
              np.int32(size), np.float32(extent)))
    return image, labels
