"""Batch all camera–LiDAR associations and robust box reductions on CUDA."""
import cupy as cp
import numpy as np

_ASSOCIATE=cp.RawKernel(r'''
extern "C" __global__ void associate(const float* ego,const float* pixels,
 const float* boxes,unsigned char* masks,int* counts,int n){
 int d=blockIdx.x,t=threadIdx.x;const float* b=boxes+d*5;
 int camera=(int)b[4];const float* uv=pixels+camera*n*3;
 __shared__ int bins[34];__shared__ int mode;
 if(t<34)bins[t]=0;__syncthreads();
 for(int i=t;i<n;i+=blockDim.x){
 float u=uv[i*3],v=uv[i*3+1],z=uv[i*3+2],h=ego[i*3+2];
 bool hit=u>b[0]&&u<b[2]&&v>b[1]&&v<b[3]&&z>1&&z<65&&h>.2f&&h<5;
 masks[d*n+i]=hit;
 if(hit)atomicAdd(bins+min(33,(int)(z/2)),1);
 }
 __syncthreads();
 if(t==0){int maximum=-1,index=0,total=0;for(int k=0;k<34;k++){total+=bins[k];if(bins[k]>maximum){maximum=bins[k];index=k;}}
 mode=total>=5?index*2+1:-100;}
 __syncthreads();
 for(int i=t;i<n;i+=blockDim.x){
 bool hit=masks[d*n+i]&&fabsf(uv[i*3+2]-mode)<2.2f;
 masks[d*n+i]=hit;
 if(hit)atomicAdd(counts+d,1);
 }
}
''','associate')
_GATHER=cp.RawKernel(r'''
extern "C" __global__ void gather(const float* ego,const unsigned char* masks,const int* ranks,
 float* compact,int n,int capacity,int total){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total||!masks[i])return;
 int d=i/n,p=i%n,k=ranks[i]-1;
 for(int axis=0;axis<3;axis++)compact[(d*capacity+k)*3+axis]=ego[p*3+axis];
}
''','gather')


def associate(ego,pixels,camera_detections):
    records=[(camera,d) for camera,ds in enumerate(camera_detections) for d in ds]
    semantics=cp.full(len(ego),-1,dtype=cp.int32)
    if not records:return [],semantics
    ego=cp.ascontiguousarray(ego,dtype=cp.float32)
    pixels=cp.ascontiguousarray(cp.stack([cp.stack(p,axis=1) for p in pixels]),dtype=cp.float32)
    boxes=cp.asarray([[*d['box'],camera] for camera,d in records],dtype=cp.float32)
    n,d=len(ego),len(records)
    masks=cp.empty((d,n),dtype=cp.uint8);counts=cp.zeros(d,dtype=cp.int32)
    _ASSOCIATE((d,),(256,),(ego,pixels,boxes,masks,counts,np.int32(n)))
    capacity=int(cp.max(counts))
    if capacity<5:return [],semantics
    ranks=cp.cumsum(masks,axis=1,dtype=cp.int32)
    compact=cp.full((d,capacity,3),cp.inf,dtype=cp.float32)
    _GATHER(((d*n+255)//256,),(256,),(ego,masks,ranks,compact,np.int32(n),np.int32(capacity),np.int32(d*n)))
    ordered=cp.sort(compact,axis=1)
    row=cp.arange(d,dtype=cp.int32)
    def quantile(q):
        position=cp.maximum(counts-1,0)*q;lo=cp.floor(position).astype(cp.int32)
        hi=cp.ceil(position).astype(cp.int32);weight=(position-lo)[:,None]
        low=ordered[row,lo];high=ordered[row,hi]
        return cp.where(counts[:,None]>0,low+(high-low)*weight,0)
    center=quantile(.5)
    extent=cp.clip(quantile(.9)-quantile(.1),cp.asarray([.4,.4,.5]),cp.asarray([8,8,5]))
    valid=(counts>=5)&(cp.linalg.norm(center[:,:2],axis=1)<=65)
    winning=cp.max(cp.where((masks>0)&valid[:,None],row[:,None]+1,0),axis=0)-1
    classes=cp.asarray([r[1]['class_id'] for r in records],dtype=cp.int32)
    semantics=cp.where(winning>=0,classes[cp.maximum(winning,0)],-1).astype(cp.int32)
    # One compact result transfer, rather than several synchronizations per object.
    result=cp.asnumpy(cp.concatenate((center,extent,counts[:,None],valid[:,None]),axis=1))
    fused=[]
    for (camera,detection),values in zip(records,result):
        if not values[7]:continue
        candidate={'center':np.round(values[:3],3).tolist(),'size':np.round(values[3:6],3).tolist(),
                   'yaw':0,'class_id':detection['class_id'],'label':detection['label'],
                   'score':detection['score'],'points':int(values[6]),'camera':camera,
                   'source':'TensorRT 2D detection + batched CUDA LiDAR association'}
        duplicate=next((x for x in fused if x['class_id']==candidate['class_id']
                       and np.linalg.norm(np.asarray(x['center'][:2])-values[:2])<2),None)
        if duplicate:
            if candidate['score']>duplicate['score']:duplicate.update(candidate)
        else:fused.append(candidate)
    return fused,semantics
