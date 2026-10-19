"""Calibrated camera textures on an ego ground plane; no semantic lane prediction."""
import cupy as cp
import numpy as np


_SURROUND = cp.RawKernel(r'''
extern "C" __global__ void surround(const unsigned char* const* cameras,
 const int* shapes,const float* matrices,unsigned char* output,int size,float extent){
 int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=size*size)return;
 float x=extent-(pixel/size+.5f)*2*extent/size;
 float y=extent-(pixel%size+.5f)*2*extent/size;
 float color[3]={0,0,0},total=0;
 for(int camera=0;camera<6;camera++){
  const float* p=matrices+camera*16;int height=shapes[camera*2],width=shapes[camera*2+1];
  float depth=p[8]*x+p[9]*y+p[11];if(depth<=.5f)continue;
  float u=(p[0]*x+p[1]*y+p[3])/depth,v=(p[4]*x+p[5]*y+p[7])/depth;
  if(u<2||u>=width-2||v<2||v>=height-2)continue;
  float side=(u-width*.5f)/(width*.5f),weight=fmaxf(0.f,1-side*side);
  weight*=weight;weight*=weight;weight*=weight;
  int u0=(int)u,v0=(int)v;float fu=u-u0,fv=v-v0;
  const unsigned char* image=cameras[camera];
  for(int c=0;c<3;c++){
   float value=(1-fv)*((1-fu)*image[(v0*width+u0)*3+c]+fu*image[(v0*width+u0+1)*3+c])+
     fv*((1-fu)*image[((v0+1)*width+u0)*3+c]+fu*image[((v0+1)*width+u0+1)*3+c]);
   color[c]+=value*weight;
  }
  total+=weight;
 }
 for(int c=0;c<3;c++)output[pixel*3+c]=total>1e-5f?(unsigned char)roundf(color[c]/total):(c==0?22:c==1?19:13);
 // Ego symbol only. The dark camera blind area is left unfilled.
 if(fabsf(x)<2.3f&&fabsf(y)<.95f){
  bool edge=fabsf(x)>2.12f||fabsf(y)>.8f;
  bool glass=(x>.6f&&x<1.5f)||(x<-1.f&&x>-1.65f);
  output[pixel*3]=edge?176:glass?48:205;
  output[pixel*3+1]=edge?236:glass?52:214;
  output[pixel*3+2]=edge?66:glass?35:213;
 }
}
''', 'surround')


def camera_bev(images, lidar2image, lidar2ego, size=768, extent=30.0):
    if len(images) != 6:
        raise ValueError('Six calibrated cameras are required')
    cameras = cp.asarray([image.data.ptr for image in images], dtype=cp.uint64)
    shapes = cp.asarray([image.shape[:2] for image in images], dtype=cp.int32)
    matrices = cp.asarray(np.asarray(lidar2image) @ np.linalg.inv(lidar2ego), dtype=cp.float32)
    output = cp.empty((size, size, 3), dtype=cp.uint8)
    _SURROUND(((size*size+255)//256,), (256,),
              (cameras, shapes, matrices, output, np.int32(size), np.float32(extent)))
    return output
